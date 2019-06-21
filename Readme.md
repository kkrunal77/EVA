# **GoogLeNet incarnation of the Inception architecture**

[Refance](https://arxiv.org/pdf/1409.4842.pdf)

**Where K = Kernel size, P= Padding, S= Stride**

**η_in = Number Of Input Feature, η_out = Number Of Output Feature** 

**R_in = Input receptive Field, R_out = Output Receptive field**  

**J_in = Input Jump, J_out = Output Jump(increase in jump)**



| Layers         | K    | P    | S    | η_in | η_out=<br />(η_in+2P-K)/S +1 | J_in | J_out=J_in*S | R_in | R_out=<br />R_in + (K-1)*J_in |
| -------------- | ---- | ---- | ---- | ---- | ---------------------------- | ---- | ------------ | ---- | ----------------------------- |
| convolution    | 7    | 2    | 2    | 224  | 112                          | 1    | 2            | 1    | 1+(7-1)*1=7                   |
| max_pool       | 3    | 0    | 2    | 112  | 56                           | 2    | 4            | 7    | 7+(3-1)*2=11                  |
| convolution    | 3    | 1    | 1    | 56   | 56                           | 4    | 4            | 11   | 11+(3-1)*4=19                 |
| max_pool       | 3    | 0    | 2    | 56   | 28                           | 4    | 8            | 19   | 19+(3-1)*4=27                 |
| inception (3a) | 5    | 2    | 1    | 28   | 28                           | 8    | 8            | 27   | 27+(5-1)*8=59                 |
| inception (3b) | 5    | 2    | 1    | 28   | 28                           | 8    | 8            | 59   | 59+(5-1)*8=91                 |
| max_pool       | 3    | 0    | 2    | 28   | 14                           | 8    | 16           | 91   | 91+(3-1)*8=123                |
| inception (4a) | 5    | 2    | 1    | 14   | 14                           | 16   | 16           | 123  | 123+(5-1)*16=187              |
| inception (4b) | 5    | 2    | 1    | 14   | 14                           | 16   | 16           | 187  | 187+(5-1)*16=251              |
| inception (4c) | 5    | 2    | 1    | 14   | 14                           | 16   | 16           | 251  | 251+(5-1)*16=315              |
| inception (4d) | 5    | 2    | 1    | 14   | 14                           | 16   | 16           | 315  | 315+(5-1)*16=379              |
| inception (4e) | 5    | 2    | 1    | 14   | 14                           | 16   | 16           | 379  | 379+(5-1)*16=443              |
| max_pool       | 3    | 0    | 2    | 14   | 7                            | 16   | 32           | 443  | 443+(3-1)*16=475              |
| inception (5a) | 5    | 2    | 1    | 7    | 7                            | 32   | 32           | 475  | 475+(5-1)*32=603              |
| inception (5b) | 5    | 2    | 1    | 7    | 7                            | 32   | 32           | 603  | 603+(5-1)*32=731              |
| avg pool       | 7    | 1    | 1    | 1    | 1                            | 32   | 32           | 731  | 731+(7-1)*32=923              |
| dropout (40%)  | -    | -    | -    | -    | -                            | -    | -            | -    | -                             |
| linear         | -    | -    | -    | -    | -                            | -    | -            | -    | -                             |
| softmax        | -    | -    | -    | -    | -                            | -    | -            | -    | -                             |











# Calculate Receptive Field for VGG16.

[Refrance](<http://zike.io/posts/calculate-receptive-field-for-vgg-16/>)

**Where K = Kernel size, P= Padding, S= Stride**

**η_in = Number Of Input Feature, η_out = Number Of Output Feature** 

**R_in = Input receptive Field, R_out = Output Receptive field**  

**J_in = Input Jump, J_out = Output Jump**

| Layers | K    | P    | S    | η_in | η_out=<br />(η_in+2P-K)/S +1 | J_in | J_out=J_in*S | R_in | R_out=<br />R_in + (K-1)*J_in |
| -----: | ---- | ---- | ---- | ---- | ---------------------------- | ---- | ------------ | ---- | ----------------------------- |
|   conv | 3    | 1    | 1    | 224  | (224+2-3)/1+1=224            | 1    | 1            | 1    | 1+(3-1)*1=3                   |
|   conv | 3    | 1    | 1    | 224  | (224+2-3)/1+1=224            | 1    | 1            | 3    | 3+(3-1)*1=5                   |
|   MP_1 | 2    | 0    | 2    | 224  | (224+0-2)/2+1=112            | 1    | 2            | 5    | 5+(2-1)*2=6                   |
|   conv | 3    | 1    | 1    | 112  | (112+2-3)/1+1=112            | 2    | 2            | 6    | 6+(3-1)*2=10                  |
|   conv | 3    | 1    | 1    | 112  | (112+2-3)/1+1=112            | 2    | 2            | 10   | 10+(3-1)*2=14                 |
|   MP_2 | 2    | 0    | 2    | 112  | (112+0-2)/2+1=56             | 2    | 4            | 14   | 14+(2-1)*2=16                 |
|   conv | 3    | 1    | 1    | 56   | (56+2-3)/1+1=56              | 4    | 4            | 16   | 16+(3-1)*4=24                 |
|   conv | 3    | 1    | 1    | 56   | (56+2-3)/1+1=56              | 4    | 4            | 24   | 24+(3-1)*4=32                 |
|   conv | 3    | 1    | 1    | 56   | (56+2-3)/1+1=56              | 4    | 4            | 32   | 32+(3-1)*4=40                 |
|   MP_3 | 2    | 0    | 2    | 56   | (56+0-2)/2+1=28              | 4    | 8            | 40   | 40+(2-1)*8=44                 |
|   conv | 3    | 1    | 1    | 28   | (28+2-3)/1+1=28              | 8    | 8            | 44   | 44+(3-1)*8=60                 |
|   conv | 3    | 1    | 1    | 28   | (28+2-3)/1+1=28              | 8    | 8            | 60   | 60+(3-1)*8=76                 |
|   conv | 3    | 1    | 1    | 28   | (28+2-3)/1+1=28              | 8    | 8            | 76   | 76+(3-1)*8=92                 |
|   MP_4 | 2    | 0    | 2    | 28   | (28+0-2)/2+1=14              | 8    | 16           | 92   | 92+(2-1)*8=100                |
|   conv | 3    | 1    | 1    | 14   | (14+2-3)/1+1=14              | 16   | 16           | 100  | 100+(3-1)*16=132              |
|   conv | 3    | 1    | 1    | 14   | (14+2-3)/1+1=14              | 16   | 16           | 132  | 132+(3-1)*16=164              |
|   conv | 3    | 1    | 1    | 14   | (14+2-3)/1+1=14              | 16   | 16           | 164  | 164+(3-1)*16=196              |
|   MP_5 | 2    | 0    | 2    | 14   | (14+0-2)/2+1=7               | 16   | 32           | 196  | 196+(2-1)*16=212              |
|     FC | 7    | 0    | 1    | 7    | (7+0-7)/1+1=1                | 32   | 32           | 212  | 212+(7-1)*32=404              |
|        |      |      |      |      |                              |      |              |      |                               |
|        |      |      |      |      |                              |      |              |      |                               |



[Paper referrnce](<https://arxiv.org/pdf/1409.4842.pdf>)



