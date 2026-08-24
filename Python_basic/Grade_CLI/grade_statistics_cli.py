import sys
# 读取文件并打印输出
def read(name):
    f = open(name,"r",encoding="UTF-8")
    f.readline()    #跳过第一行
    return f

def stroage(f):
    # 通过循环拿到所有信息并存入dict
    dict={} #存学生和分数
    list_sutdent=[] #只存学生

    for i in f.readlines():
        i=i.replace("\n","")
        list=i.split(",")   #使用，隔开返回list
        try:
            # 要转成int类型才能计算 但是若传进来的不是数字则会报错所以要try expact
            #还有一个要是传进来的分数是空的，那这里就会报越界访问
            score=int(list[1])  
            # 这里要判断分数是否有效0-100
            if(0<=score<=100):
                if list[0] in dict:
                    print(f"{list[0]}重复出现，使用最后一次成绩")
                dict[list[0]]=score  
            else:
                print(f"姓名{list[0]}的学生的分数无效")
                continue
        except ValueError:
            print(f"姓名{list[0]}的学生的分数不是数字")
            continue
        except IndexError:
            print(f"姓名为{list[0]}的学生缺少分数")
            continue

        # 只有成功才可以添加进列表 不超成功就直接跳过
        list_sutdent.append(list[0])
    passed=0
    unpassed=0
    for i in dict.values():
        i=int(i)    
        if(i>=60):
            passed+=1
        else:
            unpassed+=1
    return dict,passed,unpassed

def my_print(dict,passed,unpassed):     #统一使用dict做计算
    print(f"学生人数：{len(dict)}")
    print(f"平均分：{sum(dict.values())/len(dict)}")
    print(f"最高分：{max(dict.values())}")
    print(f"最低分：{min(dict.values())}")
    print(f"及格人数：{passed}")
    print(f"不及格人数：{unpassed}")

if len(sys.argv) < 2:       #通过终端运行
    print("用法：python grade_statistics_cli.py <成绩文件路径>")
else:
    file_path = sys.argv[1]     #这个就是成绩的路径
    f = read(file_path)

    #进一步需要有防空读取  若读取的my_dict为空则再返回报错结果
    my_dict, passed, unpassed = stroage(f)
    if (len(my_dict)==0):
        print("没有有效数据")
        f.close()
    else:
        my_print(my_dict, passed, unpassed)
        f.close()
