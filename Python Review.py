import random

def main():
     print('문제1:')
     concatenate_list()
     print('---------------------------------------------------------')
     print('문제2:')
     calculateGrade()

def concatenate_list(): #문제 1
    LstA = [random.randint(1,100) for i in range(20)]
    LstB = [random.randint(1,100) for i in range(20)]
    print(list(set(LstA + LstB)))

def calculateGrade(): #문제 2
    answer = []
    stdN =[]

    for i in range(10): #질문에 대한 정답:
        a=chr(random.randrange(65,69))
        answer.append(a)

    for i in range(8): #질문에 대한 학생의 답:
        stdA = []
        for j in range(10):
            stdA.append(chr(random.randrange(65,69)))
        stdN.append(stdA)


    for i in range(8):
        o =  0
        for j in range(10):
            if stdN[i][j] == answer[j]:
                o += 1
            else:
             pass
        print(i, '번 학생의 정답 문항의 개수는' ,o,'입니다.')

main()