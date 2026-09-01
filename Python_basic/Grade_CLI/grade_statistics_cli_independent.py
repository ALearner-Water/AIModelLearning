import sys

def read(file_name):
    f=open(file_name,"r",encoding="UTF-8")
    f.readline()
    return f

def storage(f):
    #使用for循环来获取数据
    my_list=[]  #存名字
    my_dict={}  #存名字和分数
    for data in f.readlines():
        #需要去除换行符和逗号
        data=data.replace("\n","")
        data_list=data.split(",")
        try:
            score=int(data_list[1])     #转换类型这里需要try
            if(0<=score<=100):
                my_list.append(data_list[0])
                if(data_list[0] in my_dict):
                    print("该姓名学生已存在，成绩覆盖")
                my_dict[data_list[0]]=score
            else:
                print("分数不在有效区间")
                continue
        except ValueError:
            print("分数不是数字")
            continue
        except IndexError:
            print("分数数据为空")
            continue

    return my_list,my_dict

def my_print(my_dict):
    passed=0
    unpassed=0
    for i in my_dict.values():
        if(i>=60):
            passed+=1
        else:
            unpassed+=1
    print(f"有{len(my_dict)}名学生")
    print(f"最高分为{max(my_dict.values())}")
    print(f"最低分为{min(my_dict.values())}")
    print(f"平均分为{sum(my_dict.values())/len(my_dict)}")
    print(f"及格人数有{passed}")
    print(f"不及格人数有{unpassed}")

def my_path():
    if(len(sys.argv)<2):
        print("请输入路径")
    else:
        path=sys.argv[1]
        try:    #文件可能不存在
            f=read(path)
        except FileNotFoundError:
            print("文件不存在")
        else:
            my_list,my_dict=storage(f)
            if(len(my_list)==0):
                print("数据为空")
            else: 
                my_print(my_dict)

my_path()