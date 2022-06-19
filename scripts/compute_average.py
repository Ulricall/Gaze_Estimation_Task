if __name__ == '__main__':
    path_1 = "../outputs_1.txt"
    path_2 = "../outputs_2_MPII.txt"
    path_3 = "../outputs_2_Columbia.txt"
    
    outputs_1 = open(path_1, encoding = "utf-8").readlines()
    outputs_2 = open(path_2, encoding = "utf-8").readlines()
    outputs_3 = open(path_3, encoding = "utf-8").readlines()
    
    all_sum_1 = 0
    count_1 = 0
    for item in outputs_1:
        if(item == "Start testing......\n"):
            continue
        else:
            line = item.strip("[").strip().strip("]").split(" ")
            for s in line:
                if(s == ''):
                    continue
                else:
                    all_sum_1 += float(s)
                    count_1 += 1
    average_1 = all_sum_1 / count_1
    
    all_sum_2 = 0
    count_2 = 0
    for item in outputs_2:
        if(item == "Start testing......\n"):
            continue
        else:
            line = item.strip("[").strip().strip("]").split(" ")
            for s in line:
                if(s == ''):
                    continue
                else:
                    all_sum_2 += float(s)
                    count_2 += 1
    average_2 = all_sum_2 / count_2
    
    all_sum_3 = 0
    count_3 = 0
    for item in outputs_3:
        if(item == "Start testing......\n"):
            continue
        else:
            line = item.strip("[").strip().strip("]").split(" ")
            for s in line:
                if(s == '' or s == "nan"):
                    continue
                else:
                    all_sum_3 += float(s)
                    count_3 += 1
    average_3 = all_sum_3 / count_3
    
    print("Average of output_1 is {0}".format(average_1))
    print("Average of output_2_MPII is {0}".format(average_2))
    print("Average of output_2_Columbia is {0}".format(average_3))