# 补充 sys的知识
import sys
print(sys.argv)
# [
#     "D:\\53507\\PythonAIModelLearning\\Python_basic\\Grade_CLI\\argv_demo.py",
#     "apple",
#     "123",
# # ]
# sys.argv[0]：程序自身的启动名称，可能是文件名，也可能是路径；
# sys.argv[1]：用户传入的第一个参数；
# sys.argv[2]：用户传入的第二个参数；
# # python 本身不会出现在 sys.argv 中。
