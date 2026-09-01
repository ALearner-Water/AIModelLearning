import sys
# 需要打开文件 但是文件名多变 可用函数封装
def my_open (name):
    f=open(name,"r",encoding="UTF-8")
    f.readline()     #跳过第一行
    return f
def my_read(f):
    # 将姓名存入list 将姓名和分数存入dict
    list=[]
    dict={}
    for i in f.readlines():
        i=i.replace("\n","")
        replace_list=i.split(",")
        # 分数是字符串得强制转换 有一个异常需要抛出
        try:
            score=int(replace_list[1])
            if(0<=score<=100):
                if replace_list[0] in dict:
                    print(f"{replace_list[0]}重复出现，使用最后一次成绩")
                dict[replace_list[0]]=score
                list.append(replace_list[0])
            else:
                print("数据无效，无法读取计算")
        except ValueError:
            print("不是数字，无法读取计算")
            continue
        except IndexError:
            print("数据为空，无法读取计算")
            continue

    return list,dict
def my_print(my_dict:dict):
    unpass=0
    passed=0
    for i in my_dict.values():
        if(i>=60):
            passed+=1
        else:
            unpass+=1
    print(f"有{len(my_dict)}位学生")
    print(f"平均分为{sum(my_dict.values())/len(my_dict)}")
    print(f"不及格有{unpass}人")
    print(f"及格有{passed}人")
    print(f"最高分为{max(my_dict.values())}")
    print(f"最低分为{min(my_dict.values())}")
def end():
    if (len(sys.argv)<2):
        print("路径无效")
    else:
        path=sys.argv[1]
        try:
            f=my_open(path)
        except FileNotFoundError:
            print("文件不存在")
        else:
            my_list,my_dict=my_read(f)
            if(len(my_list)==0):
                print("数据为空")
                f.close()
            else:
                my_print(my_dict)
                f.close()