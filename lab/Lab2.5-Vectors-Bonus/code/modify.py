import os
if __name__=="__main__":
    target=[]
    with open("multi.cpp",'r',encoding='utf-8') as f:
        line=f.readline()
        while line:
            if line.find("可以修改的代码区域")!=-1:
                break
            line=f.readline()
        if line:
            line=f.readline()
            # with open("base.cpp",'w') as f1:
            line=f.readline()
            
            while line and line.find("// -----------------------------------")==-1:
                target.append(line)
                line=f.readline()
    with open("base.cpp",'r',encoding='utf-8') as f:
        new_lines=f.readlines()
    with open("base.cpp",'w',encoding='utf-8') as f:
        new_lines[56:56]=target
        f.writelines(new_lines)
    # print(target)