# 不同于其他语言Python的列表可以放任意类型的元素，但是一般还是存一样的数据类型
list=[1,"nihao",2.0,"hello"]
# print(list,sep="   ") 直接加分隔符不能在列表里面进行分割，因为列表是一个整体，但是可以使用*解包进行拆分
print(*list,sep="   ")
list2=[]+15*[3] #创建一个长度为15，数据都为3的列表
print(list2)
# 遍历列表打印
for item in list:   #in是直接取出元素，range是取出索引  range(len(list))是取出索引范围
    if type(item)==str:
        print(item,end="  ")
    else:
        print("fuck",end="  ")
print()        

# 使用range写循环
for i in range(len(list)):
    print(list[i],end="  ")        