# onnx-infer-time-android

可以非常方便的测出该onnx模型在该安卓设备上的推理耗时

## 使用步骤
- 将onnx模型的模型名称修改为 'text.onnx' 并放入'app\src\main\assets\models'
- 修改'app\src\main\java\com\akuvox\speech\MainActivity.java'中关于模型推理输入的部分，下面举一个例子：

输入配置

以目录下的test.onnx为例，他是gtcrn的模型，共有4个输入：
```
name: mix
tensor: float32[1,257,1,2]

name: conv_cache
tensor: float32[2,1,16,16,33]

name: tra_cache
tensor: float32[2,3,1,1,16]

name: inter_cache
tensor: float32[2,1,33,16]
```
我们修改java文件中对应的tensor如下
``` java
//首先创建全空的数组，因为输入都是float32，所以数组数据类型也是float，如果输入是int64，那么需要创建long类型的数组
float[][][][] inputSpeech = new float[1][257][1][2];
float[][][][][]  x1 = new float[2][1][16][16][33];
float[][][][][] x2 = new float[2][3][1][1][16];4
float[][][][] x3 = new float[2][1][33][16];
//必要时可以给数组进行赋值，一般来说全0就可以

//创建tensor
OnnxTensor inputFrameTensor = OnnxTensor.createTensor(env, inputSpeech);
OnnxTensor x1Tensor = OnnxTensor.createTensor(env, x1);
OnnxTensor x2Tensor = OnnxTensor.createTensor(env, x2);
OnnxTensor x3Tensor = OnnxTensor.createTensor(env, x3);

//创建输入的列表，并把上步的tensor添加进去
Map<String, OnnxTensor> inputs = new HashMap<>();
inputs.put("mix", inputFrameTensor);
inputs.put("conv_cache", x1Tensor);
inputs.put("tra_cache", x2Tensor);
inputs.put("inter_cache", x3Tensor);
```
- 编译
使用Android Studio进行build，将生成的app在设备上安装，进入app后点击开始按钮，推理完毕后会输出单次推理耗时（3次预热推理之后，连续推理20次取平均，有需要可以修改）


# English

Easily measure the inference time of the ONNX model on the Android device.

## Steps to Use
- Rename your ONNX model to 'text.onnx' and place it in 'app\src\main\assets\models'.
- Modify the input section for model inference in 'app\src\main\java\com\akuvox\speech\MainActivity.java'. Below is an example:

Input Configuration

Take 'test.onnx' in the directory as an example. It is a GTCRN model with four inputs:
```
name: mix
tensor: float32[1,257,1,2]

name: conv_cache
tensor: float32[2,1,16,16,33]

name: tra_cache
tensor: float32[2,3,1,1,16]

name: inter_cache
tensor: float32[2,1,33,16]
```
We modify the corresponding tensors in the Java file as follows:
``` java
// First, create empty arrays. Since all inputs are float32, the array data type is also float. If the input is int64, then you need to create arrays of type long.
float[][][][] inputSpeech = new float[1][257][1][2];
float[][][][][] x1 = new float[2][1][16][16][33];
float[][][][][] x2 = new float[2][3][1][1][16];
float[][][][] x3 = new float[2][1][33][16];
// Assign values to the arrays if necessary; generally, setting them to zero is sufficient.

// Create tensors
OnnxTensor inputFrameTensor = OnnxTensor.createTensor(env, inputSpeech);
OnnxTensor x1Tensor = OnnxTensor.createTensor(env, x1);
OnnxTensor x2Tensor = OnnxTensor.createTensor(env, x2);
OnnxTensor x3Tensor = OnnxTensor.createTensor(env, x3);

// Create a list of inputs and add the tensors from the previous step
Map<String, OnnxTensor> inputs = new HashMap<>();
inputs.put("mix", inputFrameTensor);
inputs.put("conv_cache", x1Tensor);
inputs.put("tra_cache", x2Tensor);
inputs.put("inter_cache", x3Tensor);
```
- Compile：Use Android Studio to build the project. Install the generated app on your device and enter the app. After clicking the start button, it will output the inference time for a single run (averaged over 20 consecutive inferences after 3 warm-up runs; this can be modified if needed).