#include <jni.h>
#include <vector>
#include <memory>
#include "ByteTrack/BYTETracker.h"

// 为方便管理跟踪器对象，使用指针
static std::unique_ptr<byte_track::BYTETracker> g_tracker = nullptr;

extern "C" {

// -----------------------------
// 初始化跟踪器
// Java 调用：
// initTracker(int fps, int trackBuffer)
// -----------------------------
JNIEXPORT void JNICALL
Java_com_example_bytetrack_ByteTrackJni_initTracker(JNIEnv* env, jobject thiz,
                                                    jint fps, jint trackBuffer) {
    g_tracker = std::make_unique<byte_track::BYTETracker>(fps, trackBuffer);
}

// -----------------------------
// 输入一帧检测结果
// detections: float[] 每4个元素为 x, y, width, height
// probs: float[] 每个目标的置信度
// 输出: int[] 每个目标的 track_id
//        float[] 每个目标的矩形框 (x, y, width, height)
// -----------------------------
JNIEXPORT jobject JNICALL
Java_com_example_bytetrack_ByteTrackJni_updateTracker(JNIEnv* env, jobject thiz,
                                                      jfloatArray detections,
                                                      jfloatArray probs) {
    if (!g_tracker) return nullptr;

    jsize numElements = env->GetArrayLength(detections);
    jsize numProbs = env->GetArrayLength(probs);

    if (numElements % 4 != 0 || numElements / 4 != numProbs) return nullptr;

    jfloat* detPtr = env->GetFloatArrayElements(detections, nullptr);
    jfloat* probPtr = env->GetFloatArrayElements(probs, nullptr);

    std::vector<byte_track::Object> objs;
    for (int i = 0; i < numProbs; ++i) {
        float x = detPtr[i*4 + 0];
        float y = detPtr[i*4 + 1];
        float w = detPtr[i*4 + 2];
        float h = detPtr[i*4 + 3];
        float prob = probPtr[i];
        objs.emplace_back(byte_track::Rect(x, y, w, h), 0, prob);
    }

    auto outputs = g_tracker->update(objs);

    jclass floatArrayClass = env->FindClass("[F");
    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListCtor = env->GetMethodID(arrayListClass, "<init>", "()V");
    jmethodID arrayListAdd = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");

    jobject outList = env->NewObject(arrayListClass, arrayListCtor);

    for (auto& obj : outputs) {
        // float[]: [track_id, prob, x, y, w, h]
        jfloat tmp[6];
        tmp[0] = static_cast<float>(obj->getTrackId());  // track_id
        tmp[1] = obj->getScore();                        // prob
        tmp[2] = obj->getRect().x();
        tmp[3] = obj->getRect().y();
        tmp[4] = obj->getRect().width();
        tmp[5] = obj->getRect().height();

        jfloatArray arr = env->NewFloatArray(6);
        env->SetFloatArrayRegion(arr, 0, 6, tmp);
        env->CallBooleanMethod(outList, arrayListAdd, arr);
        env->DeleteLocalRef(arr);
    }

    env->ReleaseFloatArrayElements(detections, detPtr, 0);
    env->ReleaseFloatArrayElements(probs, probPtr, 0);

    return outList;
}

// -----------------------------
// 清理跟踪器
// -----------------------------
JNIEXPORT void JNICALL
Java_com_example_bytetrack_ByteTrackJni_releaseTracker(JNIEnv* env, jobject thiz) {
    g_tracker.reset();
}

} // extern "C"



// 对应的 JAVA 接口
//package com.example.bytetrack;
//import java.util.List;
//public class ByteTrackJni {
//    static {
//        System.loadLibrary("bytetrack"); // libbytetrack.so
//    }
//public native void initTracker(int fps, int trackBuffer);
//public native List<float[]> updateTracker(float[] detections, float[] probs);
//public native void releaseTracker();
//}


// 对应的 JAVA 调用示例
//ByteTrackJni tracker = new ByteTrackJni();       // 跟踪器，注意非多线程安全
//tracker.initTracker(30, 30);                     // fps：视频帧率（用于卡尔曼滤波预测），trackBuffer：目标丢失多少帧后删除轨迹
//float[] dets = {x1, y1, w1, h1, x2, y2, w2, h2}; // 每帧检测结果目标边框， xywh、xywh、xywh ..（注意这里输入非归一化后的位置坐标信息）
//float[] probs = {0.9f, 0.8f};                    // 每帧检测结果目标置信度：score、score ..
//List<float[]> tracks = tracker.updateTracker(dets, probs);   // 输出跟踪后的id、置信度和边框
//for (float[] t : tracks) {
//    int trackId = (int) t[0];     // id
//    float prob = t[1];            // 置信度
//    float x = t[2], y = t[3], w = t[4], h = t[5]; // 边框
//}
//tracker.releaseTracker();                         // 销毁跟踪器
