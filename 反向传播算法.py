# coding:utf-8
import random
import math
# 通过3个类控制神经网络
# 对权重矩阵进行了更新


class NeuronCaculate:  # 最底层类，主要控制各个神经元的各种计算工作

    def __init__(self,bias): # 将权重矩阵和偏置值，输入矩阵，输出值作为私有成员
        self.bias = bias
        self.weights = []

    def calculate_outputs(self, inputs): #输出由权重乘以输入，最后经过sigmoid函数计算得出
        self.inputs = inputs
        self.output = self.sigmiod(self.linear_combining())
        return self.output

    def linear_combining(self):  # 将权重和输入进行线性合并，结果作为sigmiod函数的变量值
        sum = 0
        for i in range(len(self.inputs)):
            sum += self.inputs[i]*self.weights[i]
        return float(sum+self.bias)  # 强制类型转换，返回浮点数

    def sigmiod(self,x):  #  sigmoid function
        return 1/(1+math.exp(-x))

    def hardlimit(self,x): # hardlimit function# 硬限函数，只输出0和1
        if x >= 0:
            return 1
        else:
            return 0

    def error(self, target):  # 计算误差
        return 0.5*(target-self.output)**2

    def differential_coefficient_error_to_output(self, target): #总误差到输出求导，由总误差公式可得
        return -(target-self.output)

    def differential_coefficient_output_to_input(self):  # 对sigmoid 函数进行求导数
        return self.output*(1-self.output)

    def differential_coefficient_input_weight(self,index): # 对权重进行求导 (wi*inputs[i])对wi求导为inputs[i]
        return self.inputs[index]

    def differential_coefficient_error_to_input(self,target): # 总误差对输入求导，链式法则，从总误差到输出，再从输出到输入
        return self.differential_coefficient_error_to_output(target)*\
               self.differential_coefficient_output_to_input()

class NeuronLayer:    # 构建层次网络类，利用neurons_calculate 类，管理并计算每一个神经元的输出值
    def __init__(self, num_neurons, bias):   # 每一层的神经元个数,以及他们的偏移值
        self.bias = bias if bias else random.random()   # 给bias 赋值为一个不为0的数
        self.neurons = []
        for i in range(num_neurons):  # 为每一个神经元都赋予一个计算类
            self.neurons.append(NeuronCaculate(bias))

    def Print(self):  # 打印相应的信息
        print("Neurons: ",len(self.neurons))  # 输出当前层的神经元个数
        for i in range(len(self.neurons)):
            print("Neuron: ",i+1)  # 从1号神经元开始编号
            for j in range(len(self.neurons[i].weights)):
                print('  weight: ',self.neurons[i].weights[j])
                print(' output:',self.neurons[i].output)
            print(' bias:',self.bias)

    def feedForward(self, inputs):  # 前馈层处理
        # 将输入向量进行处理，返回经过处理的数据
        outputs = []
        for neuron in self.neurons:   # 每个神经元对应的数据都要进行计算
            outputs.append(neuron.calculate_outputs(inputs))
        return outputs

    def return_outputs(self):  # 返回当前层的输出矩阵
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs



class BPNN:  # 整个神经网络的控制类
    learning_rate = 0.5  # 调控学习率，输入层，输出层，隐含层

    def __init__(self, num_inputs, num_hideLayer, num_outputs, hideLayer_weight=None, hideLayer_bias=None,
                 output_layer_weight=None, output_layer_bias=None):
        self.num_inputs = num_inputs  # 输入层的数
        self.hideLayer = NeuronLayer(num_hideLayer, hideLayer_bias)  # 构建隐含层
        self.output_Layer = NeuronLayer(num_outputs, output_layer_bias)  # 构建输出层
        self.init_weights_from_hideLayer_to_outputLayer(output_layer_weight)  #初始化输出层的权重矩阵
        self.init_weights_from_inputs_to_hideLayer(hideLayer_weight)   # 初始化隐含层的权重矩阵

    def init_weights_from_inputs_to_hideLayer(self, hideLayer_weights):  # 采用输入加随机数的算法实现矩阵初始化，目的是为了防止初始时值被赋为0
        weight_num = 0  #元素个数
        for h in range(len(self.hideLayer.neurons)): #隐含层的神经元个数
            for i in range(self.num_inputs): #每个神经元对应的输入数量
                if not hideLayer_weights:  # 如果初始的权重被设为 0 ，则随机产生一个非0元素
                    self.hideLayer.neurons[h].weights.append(random.random())
                else:
                    self.hideLayer.neurons[h].weights.append(hideLayer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hideLayer_to_outputLayer(self, outputs_weights):
        weight_num = 0
        for i in range(len(self.output_Layer.neurons)):  #输出层神经元个数
            for j in range(len(self.hideLayer.neurons)): # 隐含层神经元个数
                if not outputs_weights:
                    self.output_Layer.neurons[i].weights.append(random.random())
                else:
                    self.output_Layer.neurons[i].weights.append(outputs_weights[j])
            weight_num += 1

    def Print(self):  #打印各层的信息
        print('Hidden Layer')
        self.hideLayer.Print()
        print('------')
        print('* Output Layer')
        self.output_Layer.Print()
        print('------\n')

    def feedForward(self, inputs):  # 先计算隐含层的输出矩阵，再把隐含层的输出矩阵当做 输出层(output_layer)的输入值
        hide_Layer_outputs = self.hideLayer.feedForward(inputs)
        return self.output_Layer.feedForward(hide_Layer_outputs)

    def calculate_error(self, training_sets):  # 计算总误差
        total_error = 0
        for i in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[i]  #列表赋值
            self.feedForward(training_inputs) #根据输入矩阵 计算出当前的权重矩阵下的 输出矩阵
            for j in range(len(training_outputs)):
                total_error += self.output_Layer.neurons[j].error(training_outputs[j])
        return total_error

    def train(self, training_inputs, training_outputs):  #训练函数
        self.feedForward(training_inputs)   #先计算出实践的输出矩阵，后面与标准矩阵进行比较
        # 输出层
        differential_coefficient_error_to_input = [0] * len(self.output_Layer.neurons)  #Python列表的初始化赋值方式
        for i in range(len(self.output_Layer.neurons)):
            differential_coefficient_error_to_input[i] = self.output_Layer.neurons[
                i].differential_coefficient_error_to_input(training_outputs[i])  # 计算输出层的每个神经元，总误差对输入的偏导
        # 隐含层
        differential_coefficient_error_to_hideLayer = [0] * len(self.hideLayer.neurons)
        for i in range(len(self.hideLayer.neurons)):
            d_error_wrt_hidden_neuron_output = 0
            for j in range(len(self.output_Layer.neurons)):
                d_error_wrt_hidden_neuron_output += differential_coefficient_error_to_input[j] * \
                                                    self.output_Layer.neurons[i].weights[i]  #

                differential_coefficient_error_to_hideLayer[j] = d_error_wrt_hidden_neuron_output * \
                                                                 self.hideLayer.neurons[
                                                                     i].differential_coefficient_output_to_input()
                # 间接求出对隐含层输出的偏导

        # 更新输出层权重
        for i in range(len(self.output_Layer.neurons)):
            for j in range(len(self.output_Layer.neurons[i].weights)):
                pd_error_wrt_weight = differential_coefficient_error_to_input[i] * \
                                      self.hideLayer.neurons[i].differential_coefficient_input_weight(j)
                self.output_Layer.neurons[i].weights[j] -= self.learning_rate * pd_error_wrt_weight

        # 更新隐含层权重
        for i in range(len(self.hideLayer.neurons)):
            for j in range(len(self.hideLayer.neurons[i].weights)):
                pd_error_wrt_weight = differential_coefficient_error_to_hideLayer[i] * \
                                      self.hideLayer.neurons[i].differential_coefficient_input_weight(j)
                self.hideLayer.neurons[i].weights[j] -= pd_error_wrt_weight * self.learning_rate



