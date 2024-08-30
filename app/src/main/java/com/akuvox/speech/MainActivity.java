package com.akuvox.speech;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;

import android.text.TextUtils;
import android.util.Base64;
import android.util.Log;
import android.view.View;

import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "ONNXDEBUG";
    private Button button10;
    private Context mContext;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mContext = this;
        button10 = findViewById(R.id.button10);
        TextView inferenceTimeTextView = findViewById(R.id.inferenceTimeTextView);

        button10.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onClick(View v) {
                try {
                    Log.i(TAG,"start load");
                    // Create OrtEnvironment
                    OrtEnvironment env = OrtEnvironment.getEnvironment();
                    OrtSession.SessionOptions options = new OrtSession.SessionOptions();
                    options.setInterOpNumThreads(1);
                    options.addCPU(true);
                    Log.i(TAG, "load model file");

// 首先，将模型从 assets 目录复制到内部存储中
                    copyAssets("models"); // 假设你的模型在 assets/models 目录中
                    Log.i(TAG, "copy model file");
// 定义内部存储中模型文件的位置

                    String modelPath = mContext.getFilesDir().getAbsolutePath()+"/test.onnx";
                    File modelFile = new File(modelPath);
                    if (!modelFile.exists()) {
                        Log.e(TAG, "Model file not found in internal storage.");
                        return;
                    } else {
                        Log.i(TAG, "Model file found: " + modelFile.getAbsolutePath());
                    }


// 从内部存储中读取 ONNX 模型文件
                    InputStream inputStream = null;
                    ByteArrayOutputStream buffer = null;
                    try {
                        inputStream = new FileInputStream(modelFile);
                        buffer = new ByteArrayOutputStream();
                        int nRead;
                        byte[] data = new byte[327680]; // 可以增加缓冲区大小，如 32768
                        while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
                            buffer.write(data, 0, nRead);
                        }
                        buffer.flush();
                    } catch (IOException e) {
                        Log.e(TAG, "Error reading ONNX model file: " + e.getMessage());
                        return;
                    } finally {
                        if (inputStream != null) {
                            try {
                                inputStream.close();
                            } catch (IOException e) {
                                Log.e(TAG, "Error closing InputStream: " + e.getMessage());
                            }
                        }
                        if (buffer != null) {
                            try {
                                buffer.close();
                            } catch (IOException e) {
                                Log.e(TAG, "Error closing ByteArrayOutputStream: " + e.getMessage());
                            }
                        }
                    }

                    Log.i(TAG,"load model file2");
                    byte[] modelData = buffer.toByteArray();

                    OrtSession session = env.createSession(modelData, options);
                    Log.i(TAG,"load model");

                    // Create input tensors
                    // 创建形状为 [1, 500, 80] 的全1张量
                    float[][][][] inputSpeech = new float[1][257][1][2];
                    float[][][][][]  x1 = new float[2][1][16][16][33];
                    float[][][][][] x2 = new float[2][3][1][1][16];
//                    float x2 = 0;
                    float[][][][] x3 = new float[2][1][33][16];
                    long[][] x4 = new long[1][200];
                    float[][][] x5 = new float[1][1024][200];
                    float[][][] x6 = new float[1][768][200];
                    float[] x7 = new float[1];
                    float[] x8 = new float[1];
                    float[] x9 = new float[1];
                    float[] x10 = new float[1];
//                    x1[0] = 500;

//                    for (int i = 0; i < 50; i++) {
//                        m1[0][0][i] = 1.0f;
//                    }


                    OnnxTensor inputFrameTensor = OnnxTensor.createTensor(env, inputSpeech);
                    OnnxTensor x1Tensor = OnnxTensor.createTensor(env, x1);
                    OnnxTensor x2Tensor = OnnxTensor.createTensor(env, x2);
                    OnnxTensor x3Tensor = OnnxTensor.createTensor(env, x3);
                    OnnxTensor x4Tensor = OnnxTensor.createTensor(env, x4);
                    OnnxTensor x5Tensor = OnnxTensor.createTensor(env, x5);
                    OnnxTensor x6Tensor = OnnxTensor.createTensor(env, x6);
                    OnnxTensor x7Tensor = OnnxTensor.createTensor(env, x7);
                    OnnxTensor x8Tensor = OnnxTensor.createTensor(env, x8);
                    OnnxTensor x9Tensor = OnnxTensor.createTensor(env, x9);
                    OnnxTensor x10Tensor = OnnxTensor.createTensor(env, x10);


                    Map<String, OnnxTensor> inputs = new HashMap<>();
                    inputs.put("mix", inputFrameTensor);
                    inputs.put("conv_cache", x1Tensor);
                    inputs.put("tra_cache", x2Tensor);
                    inputs.put("inter_cache", x3Tensor);
//                    inputs.put("langids", x4Tensor);
//                    inputs.put("bert", x5Tensor);
//                    inputs.put("jabert", x6Tensor);
//                    inputs.put("noise_scale", x7Tensor);
//                    inputs.put("length_scale", x8Tensor);
//                    inputs.put("noise_scale_w", x9Tensor);
//                    inputs.put("sdp_ratio", x10Tensor);


                    OrtSession.Result ortOutputs;
                    ortOutputs = session.run(inputs);
                    ortOutputs = session.run(inputs);
                    ortOutputs = session.run(inputs);//预热
                    Log.i(TAG,"begin infer");
                    long totalTime = 0;
                    for (int i = 0; i < 20; i++) {
                        long startTime = System.currentTimeMillis();
                        ortOutputs = session.run(inputs);
                        long endTime = System.currentTimeMillis();
                        long inferenceTime = endTime - startTime;
                        totalTime += inferenceTime;
                    }

                    // Calculate average inference time
                    long averageTime = totalTime / 20;
                    inferenceTimeTextView.setText("Inference Time: " + averageTime + " ms");
                    Log.i(TAG,"Average inference time: " + averageTime + " ms");

                } catch (OrtException e) {
                    e.printStackTrace();
                    Log.e(TAG, "Error during ONNX model execution: " + e.getMessage());
                }
            }
            private void copyAssets(String path) {
                AssetManager assetManager = mContext.getAssets();
                String[] files = null;
                try {
                    // 获取指定assets目录下的所有文件和目录名
                    files = assetManager.list(path);
                } catch (IOException e) {
                    Log.e(TAG, "Failed to get asset file list.", e);
                }
                if (files != null) {
                    for (String filename : files) {
                        InputStream in = null;
                        OutputStream out = null;
                        try {
                            // 检查文件还是文件夹
                            if (assetManager.list(path + "/" + filename).length > 0) {
                                File dir = new File(mContext.getFilesDir(), filename);
                                if (!dir.exists()) {
                                    dir.mkdir();
                                }
                                copyAssets(path + "/" + filename);
                            } else {
                                // 是文件，进行复制
                                File outFile = new File(mContext.getFilesDir(), filename);
                                if (!outFile.exists()) { // 只有当文件不存在时才进行复制
                                    in = assetManager.open(path + "/" + filename);
                                    out = new FileOutputStream(outFile);
                                    copyFile(in, out);
                                    Log.i(TAG, "Copied " + filename + " to internal storage");
                                } else {
                                    Log.i(TAG, "File " + filename + " already exists, skipping copy.");
                                }
                            }
                        } catch(IOException e) {
                            Log.e(TAG, "Failed to copy asset file: " + filename, e);
                        } finally {
                            try {
                                if (in != null) in.close();
                                if (out != null) out.close();
                            } catch (IOException e) {
                                Log.e(TAG, "Failed to copy asset file catch: " + filename, e);
                            }
                        }
                    }
                }
            }

            private void copyFile(InputStream in, OutputStream out) throws IOException {
                byte[] buffer = new byte[1024];
                int read;
                while((read = in.read(buffer)) != -1){
                    out.write(buffer, 0, read);
                }
            }
        });
    }
}