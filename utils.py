#!/bin/sh
"exec" "`dirname $0`/venv/bin/python" "$0" "$@"
def IOU(box1, box2):
    #p1,p2,p3,p4-координаты узлов нижнего уровня квадрата, начиная от нижнего левого узла по часово стрелке
    #p5,p6,p7,p8-координаты узлов верхнего уровня квадрата, начиная от нижнего левого узла по часово стрелке
    #box1=[p1,p2,p3,p4,p5,p6,p7,p8]
    #box2=[l1,l2,l3,l4,l5,l6,l7,l8]
    #t1,b1,l1,r1- углы на "нижнем" уравне квадрата
    # t12,b12,l12,r12- на "верхнем" уравне квадрата
    p1,p2,p3,p4,p5,p6,p7,p8 = box1
    l1,l2,l3,l4,l5,l6,l7,l8 = box2
    inter_w=(p3[0]-l1[0])
    inter_h=(p2[1]-l1[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0
    iner_area= inter_w*inter_h
    inter_volume=iner_area*(p5[2]-l1[2])
    if inter_volume<=0:
        return 0
    volume1=(p5[2]-p1[2])*(p2[1]-p1[1]*(p4[0]-p1[0]))
    volume2 =(l5[2] - l1[2])* (l2[1] - l1[1] * (l4[0] - l1[0]) )
    iou = float(inter_volume) / float(volume1 + volume2 - inter_volume)
    return iou

def abs_ident():
    box1=[[0,0,0],
      [0,1,0],
      [1,1,0],
      [1,0,0],
      [0, 0, 1],
      [0, 1, 1],
      [1, 1, 1],
      [1, 0, 1]
      ]
    box2=[[0,0,0],
      [0,1,0],
      [1,1,0],
      [1,0,0],
      [0, 0, 1],
      [0, 1, 1],
      [1, 1, 1],
      [1, 0, 1]
      ]
    res=IOU(box1,box2)
    print("Abs test",res)
def one_third_res():
    box1=[[0,0,0],
      [0,1,0],
      [1,1,0],
      [1,0,0],
      [0, 0, 1],
      [0, 1, 1],
      [1, 1, 1],
      [1, 0, 1]
      ]
    box2=[[0.5,0,0],
      [0.5,1,0],
      [1.5,1,0],
      [1.5,0,0],
      [0.5, 0, 1],
      [0.5, 1, 1],
      [1.5, 1, 1],
      [1.5, 0, 1]
      ]
    res=IOU(box1,box2)
    print("1/3 Test",res)

def no_ident():
    box1=[[0,0,0],
      [0,1,0],
      [1,1,0],
      [1,0,0],
      [0, 0, 1],
      [0, 1, 1],
      [1, 1, 1],
      [1, 0, 1]
      ]
    box2=[[3.5,0,0],
      [3.5,10,0],
      [4.5,10,0],
      [3.5,0,0],
      [3.5, 0, 10],
      [3.5, 10, 10],
      [4.5, 10, 10],
      [4.5, 0, 10]
      ]
    res=IOU(box1,box2)
    print("no ident",res)



abs_ident()
one_third_res()
no_ident()
