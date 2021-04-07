""" Rank-5 tensors from Mathieu's Mathematica code.
    First index is the one to contract
    S = 0 tensors: T_40_A1_1, T_40_A1_2, T_22_A1_1, T_22_A1_2, T_04_A1_1
    S = 1 tensors: T_31_A1_1, T_31_A1_2, T_13_A1_1"""

import torch

# S=0, n_occ=(4,0)

T_40_A1_1 = torch.zeros((4,4,4,4,4), dtype=torch.float64)
T_40_A1_1[0,0,0,2,2]=1/3
T_40_A1_1[0,0,1,2,1]=-1/3
T_40_A1_1[0,0,2,2,0]=1/3
T_40_A1_1[0,1,0,1,2]=-1/3
T_40_A1_1[0,1,1,1,1]=1/3
T_40_A1_1[0,1,2,1,0]=-1/3
T_40_A1_1[0,2,0,0,2]=1/3
T_40_A1_1[0,2,1,0,1]=-1/3
T_40_A1_1[0,2,2,0,0]=1/3

T_40_A1_2 = torch.zeros((4,4,4,4,4), dtype=torch.float64)
T_40_A1_2[0,0,0,2,2]=1/6 * ( 5 )**( -1/2 )
T_40_A1_2[0,0,1,1,2]=-1/2 * ( 5 )**( -1/2 )
T_40_A1_2[0,0,1,2,1]=1/3 * ( 5 )**( -1/2 )
T_40_A1_2[0,0,2,0,2]=( 5 )**( -1/2 )
T_40_A1_2[0,0,2,1,1]=-1/2 * ( 5 )**( -1/2 )
T_40_A1_2[0,0,2,2,0]=1/6 * ( 5 )**( -1/2 )
T_40_A1_2[0,1,0,1,2]=1/3 * ( 5 )**( -1/2 )
T_40_A1_2[0,1,0,2,1]=-1/2 * ( 5 )**( -1/2 )
T_40_A1_2[0,1,1,0,2]=-1/2 * ( 5 )**( -1/2 )
T_40_A1_2[0,1,1,1,1]=2/3 * ( 5 )**( -1/2 )
T_40_A1_2[0,1,1,2,0]=-1/2 * ( 5 )**( -1/2 )
T_40_A1_2[0,1,2,0,1]=-1/2 * ( 5 )**( -1/2 )
T_40_A1_2[0,1,2,1,0]=1/3 * ( 5 )**( -1/2 )
T_40_A1_2[0,2,0,0,2]=1/6 * ( 5 )**( -1/2 )
T_40_A1_2[0,2,0,1,1]=-1/2 * ( 5 )**( -1/2 )
T_40_A1_2[0,2,0,2,0]=( 5 )**( -1/2 )
T_40_A1_2[0,2,1,0,1]=1/3 * ( 5 )**( -1/2 )
T_40_A1_2[0,2,1,1,0]=-1/2 * ( 5 )**( -1/2 )
T_40_A1_2[0,2,2,0,0]=1/6 * ( 5 )**( -1/2 )

# S=0, n_occ=(2,2)

T_22_A1_1 = torch.zeros((4,4,4,4,4), dtype=torch.float64)
T_22_A1_1[0,0,3,2,3]=( 6 )**( -1/2 )
T_22_A1_1[0,1,3,1,3]=-1 * ( 6 )**( -1/2 )
T_22_A1_1[0,2,3,0,3]=( 6 )**( -1/2 )
T_22_A1_1[0,3,0,3,2]=( 6 )**( -1/2 )
T_22_A1_1[0,3,1,3,1]=-1 * ( 6 )**( -1/2 )
T_22_A1_1[0,3,2,3,0]=( 6 )**( -1/2 )

T_22_A1_2 = torch.zeros((4,4,4,4,4), dtype=torch.float64)
T_22_A1_2[0,0,2,3,3]=1/2 * ( 3 )**( -1/2 )
T_22_A1_2[0,0,3,3,2]=1/2 * ( 3 )**( -1/2 )
T_22_A1_2[0,1,1,3,3]=-1/2 * ( 3 )**( -1/2 )
T_22_A1_2[0,1,3,3,1]=-1/2 * ( 3 )**( -1/2 )
T_22_A1_2[0,2,0,3,3]=1/2 * ( 3 )**( -1/2 )
T_22_A1_2[0,2,3,3,0]=1/2 * ( 3 )**( -1/2 )
T_22_A1_2[0,3,0,2,3]=1/2 * ( 3 )**( -1/2 )
T_22_A1_2[0,3,1,1,3]=-1/2 * ( 3 )**( -1/2 )
T_22_A1_2[0,3,2,0,3]=1/2 * ( 3 )**( -1/2 )
T_22_A1_2[0,3,3,0,2]=1/2 * ( 3 )**( -1/2 )
T_22_A1_2[0,3,3,1,1]=-1/2 * ( 3 )**( -1/2 )
T_22_A1_2[0,3,3,2,0]=1/2 * ( 3 )**( -1/2 )

# S=0, n_occ=(0,4)

T_04_A1_1 = torch.zeros((4,4,4,4,4), dtype=torch.float64)
T_04_A1_1[0,3,3,3,3]=1

# S=1, n_occ=(3,1)

T_31_A1_1 = torch.zeros((4,4,4,4,4), dtype=torch.float64)
T_31_A1_1[0,0,0,2,3]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[0,0,0,3,2]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[0,0,1,1,3]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[0,0,1,3,1]=-3/2 * ( 35 )**( -1/2 )
T_31_A1_1[0,0,2,0,3]=1/2 * ( 35 )**( -1/2 )
T_31_A1_1[0,0,2,3,0]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[0,0,3,0,2]=1/2 * ( 35 )**( -1/2 )
T_31_A1_1[0,0,3,1,1]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[0,0,3,2,0]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[0,1,0,1,3]=-3/2 * ( 35 )**( -1/2 )
T_31_A1_1[0,1,0,3,1]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[0,1,1,0,3]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[0,1,1,3,0]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[0,1,3,0,1]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[0,1,3,1,0]=-3/2 * ( 35 )**( -1/2 )
T_31_A1_1[0,2,0,0,3]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[0,2,0,3,0]=1/2 * ( 35 )**( -1/2 )
T_31_A1_1[0,2,3,0,0]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[0,3,0,0,2]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[0,3,0,1,1]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[0,3,0,2,0]=1/2 * ( 35 )**( -1/2 )
T_31_A1_1[0,3,1,0,1]=-3/2 * ( 35 )**( -1/2 )
T_31_A1_1[0,3,1,1,0]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[0,3,2,0,0]=1/4 * ( 7/5 )**( 1/2 )

T_31_A1_1[1,0,1,2,3]=3/2 * ( 35 )**( -1/2 )
T_31_A1_1[1,0,1,3,2]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,0,2,1,3]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,0,2,3,1]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,0,3,1,2]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,0,3,2,1]=3/2 * ( 35 )**( -1/2 )
T_31_A1_1[1,1,0,2,3]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,1,0,3,2]=3/2 * ( 35 )**( -1/2 )
T_31_A1_1[1,1,1,1,3]=-2 * ( 35 )**( -1/2 )
T_31_A1_1[1,1,1,3,1]=-2 * ( 35 )**( -1/2 )
T_31_A1_1[1,1,2,0,3]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,1,2,3,0]=3/2 * ( 35 )**( -1/2 )
T_31_A1_1[1,1,3,0,2]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,1,3,1,1]=-2 * ( 35 )**( -1/2 )
T_31_A1_1[1,1,3,2,0]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,2,0,1,3]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,2,0,3,1]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,2,1,0,3]=3/2 * ( 35 )**( -1/2 )
T_31_A1_1[1,2,1,3,0]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,2,3,0,1]=3/2 * ( 35 )**( -1/2 )
T_31_A1_1[1,2,3,1,0]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,3,0,1,2]=3/2 * ( 35 )**( -1/2 )
T_31_A1_1[1,3,0,2,1]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,3,1,0,2]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,3,1,1,1]=-2 * ( 35 )**( -1/2 )
T_31_A1_1[1,3,1,2,0]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,3,2,0,1]=1/4 * ( 35 )**( -1/2 )
T_31_A1_1[1,3,2,1,0]=3/2 * ( 35 )**( -1/2 )

T_31_A1_1[2,0,2,2,3]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[2,0,2,3,2]=1/2 * ( 35 )**( -1/2 )
T_31_A1_1[2,0,3,2,2]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[2,1,1,2,3]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[2,1,1,3,2]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[2,1,2,1,3]=-3/2 * ( 35 )**( -1/2 )
T_31_A1_1[2,1,2,3,1]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[2,1,3,1,2]=-3/2 * ( 35 )**( -1/2 )
T_31_A1_1[2,1,3,2,1]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[2,2,0,2,3]=1/2 * ( 35 )**( -1/2 )
T_31_A1_1[2,2,0,3,2]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[2,2,1,1,3]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[2,2,1,3,1]=-3/2 * ( 35 )**( -1/2 )
T_31_A1_1[2,2,2,0,3]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[2,2,2,3,0]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[2,2,3,0,2]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[2,2,3,1,1]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[2,2,3,2,0]=1/2 * ( 35 )**( -1/2 )
T_31_A1_1[2,3,0,2,2]=1/4 * ( 7/5 )**( 1/2 )
T_31_A1_1[2,3,1,1,2]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[2,3,1,2,1]=-3/2 * ( 35 )**( -1/2 )
T_31_A1_1[2,3,2,0,2]=1/2 * ( 35 )**( -1/2 )
T_31_A1_1[2,3,2,1,1]=-1/4 * ( 35 )**( -1/2 )
T_31_A1_1[2,3,2,2,0]=1/4 * ( 7/5 )**( 1/2 )

T_31_A1_2 = torch.zeros((4,4,4,4,4), dtype=torch.float64)
T_31_A1_2[0,0,1,1,3]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[0,0,1,3,1]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[0,0,2,0,3]=( 7 )**( -1/2 )
T_31_A1_2[0,0,3,0,2]=( 7 )**( -1/2 )
T_31_A1_2[0,0,3,1,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[0,1,0,1,3]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[0,1,0,3,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[0,1,1,0,3]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[0,1,1,3,0]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[0,1,3,0,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[0,1,3,1,0]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[0,2,0,3,0]=( 7 )**( -1/2 )
T_31_A1_2[0,3,0,1,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[0,3,0,2,0]=( 7 )**( -1/2 )
T_31_A1_2[0,3,1,0,1]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[0,3,1,1,0]=-1/2 * ( 7 )**( -1/2 )

T_31_A1_2[1,0,1,2,3]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,0,1,3,2]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,0,2,1,3]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,0,2,3,1]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,0,3,1,2]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,0,3,2,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,1,0,2,3]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,1,0,3,2]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,1,1,1,3]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,1,1,3,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,1,2,0,3]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,1,2,3,0]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,1,3,0,2]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,1,3,1,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,1,3,2,0]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,2,0,1,3]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,2,0,3,1]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,2,1,0,3]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,2,1,3,0]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,2,3,0,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,2,3,1,0]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,3,0,1,2]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,3,0,2,1]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,3,1,0,2]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,3,1,1,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,3,1,2,0]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,3,2,0,1]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[1,3,2,1,0]=-1/2 * ( 7 )**( -1/2 )

T_31_A1_2[2,0,2,3,2]=( 7 )**( -1/2 )
T_31_A1_2[2,1,1,2,3]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[2,1,1,3,2]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[2,1,2,1,3]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[2,1,2,3,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[2,1,3,1,2]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[2,1,3,2,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[2,2,0,2,3]=( 7 )**( -1/2 )
T_31_A1_2[2,2,1,1,3]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[2,2,1,3,1]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[2,2,3,1,1]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[2,2,3,2,0]=( 7 )**( -1/2 )
T_31_A1_2[2,3,1,1,2]=-1/2 * ( 7 )**( -1/2 )
T_31_A1_2[2,3,1,2,1]=1/2 * ( 7 )**( -1/2 )
T_31_A1_2[2,3,2,0,2]=( 7 )**( -1/2 )
T_31_A1_2[2,3,2,1,1]=-1/2 * ( 7 )**( -1/2 )

# S=1, n_occ=(1,3)

T_13_A1_1 = torch.zeros((4,4,4,4,4), dtype=torch.float64)
T_13_A1_1[0,0,3,3,3]=1/2
T_13_A1_1[0,3,0,3,3]=1/2
T_13_A1_1[0,3,3,0,3]=1/2
T_13_A1_1[0,3,3,3,0]=1/2

T_13_A1_1[1,1,3,3,3]=1/2
T_13_A1_1[1,3,1,3,3]=1/2
T_13_A1_1[1,3,3,1,3]=1/2
T_13_A1_1[1,3,3,3,1]=1/2

T_13_A1_1[2,2,3,3,3]=1/2
T_13_A1_1[2,3,2,3,3]=1/2
T_13_A1_1[2,3,3,2,3]=1/2
T_13_A1_1[2,3,3,3,2]=1/2

list_S0 = [T_40_A1_1, T_40_A1_2, T_22_A1_1, T_22_A1_2, T_04_A1_1]
list_S1 = [T_31_A1_1, T_31_A1_2, T_13_A1_1]