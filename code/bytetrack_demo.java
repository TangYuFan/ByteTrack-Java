package tool.deeplearning;

import ai.onnxruntime.*;
import org.apache.commons.math3.linear.*;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @desc: ByteTrack Java 最终版本
 *        - 原版 Tracker 三阶段管理
 *        - 完整匈牙利算法 + gating
 *        - Kalman Filter 8D
 *        - Track 轨迹绘制
 *        - YOLOv5 目标检测
 */
public class bytetrack_demo {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    // -------- Detection --------
    static class Detection {
        public double x, y, w, h, conf;
        public int classId;
        public Detection(double x, double y, double w, double h, double conf, int classId) {
            this.x = x; this.y = y; this.w = w; this.h = h; this.conf = conf; this.classId = classId;
        }
        public double aspectRatio() { return w / h; }
        public double cx() { return x + w / 2.0; }
        public double cy() { return y + h / 2.0; }
    }

    // -------- Track --------
    static class Track {
        public enum State { Tentative, Active, Lost }

        public int id;
        public double x, y, a, h;
        public double vx, vy, va, vh;
        public int age=0, hitStreak=0, lost=0;
        public int classId;
        public boolean confirmed=false;
        public State state;
        public List<Point> history = new ArrayList<>();

        public Track(int id, Detection det) {
            this.id=id;
            this.x=det.cx(); this.y=det.cy();
            this.a=det.aspectRatio(); this.h=det.h;
            this.classId=det.classId;
            this.hitStreak=1;
            this.state=State.Tentative;
            history.add(new Point(x,y));
        }

        public Rect getRect() {
            int w=(int)(h*a);
            int x1=(int)(x-w/2); int y1=(int)(y-h/2);
            return new Rect(x1,y1,w,(int)h);
        }

        public void updateHistory() {
            history.add(new Point(x, y));
            if(history.size() > 50) history.remove(0);
        }
    }

    // -------- Kalman Filter (8D) --------
    static class KalmanFilter8D {
        private RealMatrix F, H, P, Q, R;

        public KalmanFilter8D() {
            F = MatrixUtils.createRealIdentityMatrix(8);
            F.setEntry(0,4,1); F.setEntry(1,5,1); F.setEntry(2,6,1); F.setEntry(3,7,1);
            H = MatrixUtils.createRealMatrix(4,8);
            for(int i=0;i<4;i++) H.setEntry(i,i,1);
            P = MatrixUtils.createRealIdentityMatrix(8).scalarMultiply(1e-2);
            Q = MatrixUtils.createRealIdentityMatrix(8).scalarMultiply(1e-3);
            R = MatrixUtils.createRealIdentityMatrix(4).scalarMultiply(1e-1);
        }

        public void predict(Track t){
            RealVector state = MatrixUtils.createRealVector(new double[]{t.x,t.y,t.a,t.h,t.vx,t.vy,t.va,t.vh});
            state = F.operate(state);
            t.x=state.getEntry(0); t.y=state.getEntry(1); t.a=state.getEntry(2); t.h=state.getEntry(3);
            t.vx=state.getEntry(4); t.vy=state.getEntry(5); t.va=state.getEntry(6); t.vh=state.getEntry(7);
            P = F.multiply(P).multiply(F.transpose()).add(Q);
        }

        public void update(Track t, Detection d){
            predict(t);
            RealVector z = MatrixUtils.createRealVector(new double[]{d.cx(),d.cy(),d.aspectRatio(),d.h});
            RealMatrix Ht = H.transpose();
            RealMatrix S = H.multiply(P).multiply(Ht).add(R);
            DecompositionSolver solver = new LUDecomposition(S).getSolver();
            RealMatrix K = P.multiply(Ht).multiply(solver.getInverse());

            RealVector state = MatrixUtils.createRealVector(new double[]{t.x,t.y,t.a,t.h,t.vx,t.vy,t.va,t.vh});
            RealVector y = z.subtract(H.operate(state));
            RealVector stateNew = state.add(K.operate(y));
            t.x=stateNew.getEntry(0); t.y=stateNew.getEntry(1);
            t.a=stateNew.getEntry(2); t.h=stateNew.getEntry(3);
            t.vx=stateNew.getEntry(4); t.vy=stateNew.getEntry(5);
            t.va=stateNew.getEntry(6); t.vh=stateNew.getEntry(7);

            RealMatrix I = MatrixUtils.createRealIdentityMatrix(8);
            P = I.subtract(K.multiply(H)).multiply(P);
        }
    }

    // -------- IOU --------
    static double iou(Detection d, Track t){
        Rect r1 = new Rect((int)d.cx()-(int)(d.w/2),(int)d.cy()-(int)(d.h/2),(int)d.w,(int)d.h);
        Rect r2 = t.getRect();
        int x1=Math.max(r1.x,r2.x), y1=Math.max(r1.y,r2.y);
        int x2=Math.min(r1.x+r1.width,r2.x+r2.width), y2=Math.min(r1.y+r1.height,r2.y+r2.height);
        int w=Math.max(0,x2-x1), h=Math.max(0,y2-y1);
        double inter=w*h, union=r1.width*r1.height+r2.width*r2.height-inter;
        return union>0?inter/union:0;
    }

    // -------- Hungarian Matcher（匈牙利算法） --------
    static class HungarianMatcher {
        private double gatingIou = 0.1;
        private double minIou = 0.3;

        public Map<Track, Detection> match(List<Track> tracks, List<Detection> dets) {
            Map<Track, Detection> matches = new HashMap<>();
            if(tracks.isEmpty() || dets.isEmpty()) return matches;
            int N = tracks.size(), M = dets.size();
            double[][] cost = new double[N][M];
            for(int i=0;i<N;i++){
                for(int j=0;j<M;j++){
                    double val = iou(dets.get(j), tracks.get(i));
                    cost[i][j] = (val < gatingIou) ? 1e6 : (1 - val);
                }
            }
            int[] assignment = hungarianAlgorithm(cost);
            for(int i=0;i<N;i++){
                int j = assignment[i];
                if(j>=0 && j<M && 1-cost[i][j] >= minIou){
                    matches.put(tracks.get(i), dets.get(j));
                }
            }
            return matches;
        }

        private int[] hungarianAlgorithm(double[][] costMatrix) {
            int nRows = costMatrix.length;
            int nCols = costMatrix[0].length;
            int n = Math.max(nRows, nCols);
            double[][] cost = new double[n][n];
            for(int i=0;i<n;i++){
                for(int j=0;j<n;j++){
                    if(i<nRows && j<nCols) cost[i][j] = costMatrix[i][j];
                    else cost[i][j] = 1e6;
                }
            }

            double[] u = new double[n];
            double[] v = new double[n];
            int[] p = new int[n];
            int[] way = new int[n];

            for(int i=1;i<n;i++){
                p[0]=i; int j0=0;
                double[] minv = new double[n]; Arrays.fill(minv,1e9);
                boolean[] used = new boolean[n];
                do {
                    used[j0]=true; int i0=p[j0]; int j1=-1; double delta=1e9;
                    for(int j=1;j<n;j++){
                        if(!used[j]){
                            double cur = cost[i0][j]-u[i0]-v[j];
                            if(cur<minv[j]){ minv[j]=cur; way[j]=j0; }
                            if(minv[j]<delta){ delta=minv[j]; j1=j; }
                        }
                    }
                    for(int j=0;j<n;j++){
                        if(used[j]) { u[p[j]]+=delta; v[j]-=delta; }
                        else minv[j]-=delta;
                    }
                    j0=j1;
                } while(p[j0]!=0);
                do { int j1=way[j0]; p[j0]=p[j1]; j0=j1; } while(j0!=0);
            }

            int[] result = new int[nRows];
            Arrays.fill(result,-1);
            for(int j=1;j<n;j++){
                if(p[j]<nRows && j<nCols) result[p[j]]=j;
            }
            return result;
        }
    }

    // -------- Tracker 三阶段 --------
    static class Tracker {
        private List<Track> tracks=new ArrayList<>();
        private KalmanFilter8D kf=new KalmanFilter8D();
        private HungarianMatcher matcher=new HungarianMatcher();
        private AtomicInteger nextId=new AtomicInteger(0);

        private int minHits=3, maxLost=15;
        private double lowIouThres=0.4;

        public List<Track> update(List<Detection> dets){
            for(Track t:tracks) kf.predict(t);

            List<Detection> high=new ArrayList<>(), low=new ArrayList<>();
            for(Detection d:dets){ if(d.conf>=0.5) high.add(d); else low.add(d); }

            Map<Track, Detection> matches = matcher.match(tracks, high);
            Set<Track> matchedT=new HashSet<>();
            Set<Detection> matchedD=new HashSet<>();
            for(Map.Entry<Track, Detection> e:matches.entrySet()){
                Track t = e.getKey();
                Detection d = e.getValue();
                kf.update(t,d);
                t.lost=0; t.hitStreak++;
                if(t.hitStreak>=minHits) t.state=Track.State.Active;
                t.updateHistory();
                matchedT.add(t); matchedD.add(d);
            }

            for(Detection d:high){
                if(!matchedD.contains(d)) tracks.add(new Track(nextId.getAndIncrement(), d));
            }

            for(Detection d:low){
                for(Track t:tracks){
                    if(t.classId!=d.classId) continue;
                    if(iou(d,t)>lowIouThres){
                        kf.update(t,d); t.lost=0; t.hitStreak++;
                        if(t.hitStreak>=minHits) t.state=Track.State.Active;
                        t.updateHistory();
                        break;
                    }
                }
            }

            Iterator<Track> it = tracks.iterator();
            while(it.hasNext()){
                Track t=it.next();
                if(!matchedT.contains(t)) t.lost++;
                if(t.lost>maxLost) it.remove();
                else if(t.state==Track.State.Active && t.lost>0) t.state=Track.State.Lost;
            }

            return tracks;
        }
    }

    // -------- YOLOv5 推理 --------
    static class Yolov5 {
        private OrtEnvironment env;
        private OrtSession session;
        private int width=640,height=640,channel=3;
        private float iou_thres=0.5f, obj_score_thres=0.3f, class_score_thres=0.25f;

        public Yolov5() {
            String model="C:\\work\\workspace\\deeplearn-java\\model\\deeplearning\\yolov5\\yolov5s.onnx";
            try { env=OrtEnvironment.getEnvironment(); session=env.createSession(model,new OrtSession.SessionOptions()); }
            catch(Exception e){ e.printStackTrace(); }
        }

        public Mat resizeWithoutPadding(Mat src,int w,int h){
            Mat dst=new Mat(); Imgproc.resize(src,dst,new Size(w,h));
            return dst;
        }

        public float[] whc2cwh(float[] src){
            float[] chw=new float[src.length]; int j=0;
            for(int ch=0;ch<3;ch++) for(int i=ch;i<src.length;i+=3) chw[j++]=src[i];
            return chw;
        }

        public float[] xywh2xyxy(float[] b,float maxW,float maxH){
            float x=b[0],y=b[1],w=b[2],h=b[3];
            float x1=x-w/2,y1=y-h/2,x2=x+w/2,y2=y+h/2;
            return new float[]{Math.max(0,x1),Math.max(0,y1),Math.min(maxW,x2),Math.min(maxH,y2)};
        }

        public double calculateIoU(float[] b1,float[] b2){
            double x1=Math.max(b1[0],b2[0]),y1=Math.max(b1[1],b2[1]);
            double x2=Math.min(b1[2],b2[2]),y2=Math.min(b1[3],b2[3]);
            double inter=Math.max(0,x2-x1+1)*Math.max(0,y2-y1+1);
            double union=(b1[2]-b1[0]+1)*(b1[3]-b1[1]+1)+(b2[2]-b2[0]+1)*(b2[3]-b2[1]+1)-inter;
            return inter/union;
        }

        // -------- Class-wise NMS + 低置信度恢复 --------
        public List<Detection> infer(Mat frame){
            List<Detection> dets=new ArrayList<>();
            int origW=frame.cols(),origH=frame.rows();
            float scaleX=(float)origW/width, scaleY=(float)origH/height;

            Mat mat=resizeWithoutPadding(frame,width,height);
            try{
                Imgproc.cvtColor(mat,mat,Imgproc.COLOR_BGR2RGB);
                mat.convertTo(mat,CvType.CV_32FC1,1.0/255.0);
                float[] whc=new float[width*height*channel];
                int idx=0;
                for(int i=0;i<height;i++){
                    float[] row=new float[width*channel];
                    mat.get(i,0,row);
                    System.arraycopy(row,0,whc,idx,row.length);
                    idx+=row.length;
                }
                float[] chw=whc2cwh(whc);

                try(OnnxTensor input=OnnxTensor.createTensor(env,FloatBuffer.wrap(chw),new long[]{1,channel,width,height});
                    OrtSession.Result res=session.run(Collections.singletonMap("images",input))){

                    Object raw=res.get(0).getValue();
                    float[][] output=(raw instanceof float[][][]) ? ((float[][][])raw)[0] : (float[][])raw;

                    // -------- 分类分组
                    Map<Integer,List<float[]>> classMap=new HashMap<>();
                    for(float[] data:output){
                        float objConf=data[4];
                        int classId=0; float maxClassConf=data[5];
                        for(int c=5;c<data.length;c++) if(data[c]>maxClassConf){maxClassConf=data[c];classId=c-5;}
                        float score=objConf*maxClassConf;
                        if(objConf>=obj_score_thres && maxClassConf>=class_score_thres){
                            float[] xyxy=xywh2xyxy(new float[]{data[0],data[1],data[2],data[3]},width,height);
                            data[0]=xyxy[0]; data[1]=xyxy[1]; data[2]=xyxy[2]; data[3]=xyxy[3];
                            data[4]=score; data[5]=classId;
                            classMap.computeIfAbsent(classId,k->new ArrayList<>()).add(data);
                        }
                    }

                    // -------- Class-wise NMS
                    for(Map.Entry<Integer,List<float[]>> entry:classMap.entrySet()){
                        List<float[]> objs=entry.getValue();
                        objs.sort((o1,o2)->Float.compare(o2[4],o1[4]));
                        ArrayList<float[]> nms=new ArrayList<>();
                        while(!objs.isEmpty()){
                            float[] max=objs.get(0); nms.add(max); objs.remove(0);
                            objs.removeIf(o->calculateIoU(max,new float[]{o[0],o[1],o[2],o[3]})>iou_thres);
                        }
                        for(float[] b:nms){
                            float x1=b[0]*scaleX, y1=b[1]*scaleY;
                            float x2=b[2]*scaleX, y2=b[3]*scaleY;
                            dets.add(new Detection(x1,y1,x2-x1,y2-y1,b[4],(int)b[5]));
                        }
                    }
                }
            }catch(Exception e){ e.printStackTrace(); }
            System.out.println("检测数量："+dets.size());
            return dets;
        }
    }

    // -------- 绘制轨迹与检测 --------
    static void drawTracks(Mat frame,List<Track> tracks){
        for(Track t:tracks){
            if(t.state!=Track.State.Active) continue;
            Rect r=t.getRect();
            Imgproc.rectangle(frame,r,new Scalar(0,255,0),2);
            Imgproc.putText(frame,"ID:"+t.id,new Point(r.x,r.y-5),
                    Imgproc.FONT_HERSHEY_SIMPLEX,0.6,new Scalar(0,255,0),2);
            for(int i=1;i<t.history.size();i++){
                Point p1 = t.history.get(i-1);
                Point p2 = t.history.get(i);
                Imgproc.line(frame, p1, p2, new Scalar(0,255,255), 2);
            }
        }
    }

    static void drawDetect(Mat frame,List<Detection> dets){
        for(Detection d:dets){
            Rect r=new Rect((int)d.x,(int)d.y,(int)d.w,(int)d.h);
            Imgproc.rectangle(frame,r,new Scalar(255,0,0),2);
            Imgproc.putText(frame,String.format("cls:%d %.2f",d.classId,d.conf),
                    new Point(d.x,d.y-5),Imgproc.FONT_HERSHEY_SIMPLEX,0.5,new Scalar(255,255,255),1);
        }
    }


    public static void main(String[] args){
        VideoCapture cap=new VideoCapture(0);
        if(!cap.isOpened()){ System.out.println("无法打开摄像头"); return; }
        Mat frame=new Mat();
        Tracker tracker=new Tracker();
        Yolov5 yolov5 = new Yolov5();
        while(cap.read(frame)){
            if(frame.empty()) break;
            List<Detection> dets= yolov5.infer(frame);
//            drawDetect(frame,dets);
            System.out.println("目标个数："+dets.size());
            List<Track> tracks=tracker.update(dets);
            drawTracks(frame,tracks);
            HighGui.imshow("ByteTrack",frame);
            HighGui.waitKey(1);
        }
        cap.release();
    }
}
