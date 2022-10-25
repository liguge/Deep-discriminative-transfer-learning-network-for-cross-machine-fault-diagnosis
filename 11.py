# parameters: 791972
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1              [-1, 32, 192]            2080
       BatchNorm1d-2              [-1, 32, 192]              64
              ReLU-3              [-1, 32, 192]               0
         MaxPool1d-4               [-1, 32, 96]               0
            Conv1d-5               [-1, 64, 96]            6208
       BatchNorm1d-6               [-1, 64, 96]             128
              ReLU-7               [-1, 64, 96]               0
         MaxPool1d-8               [-1, 64, 48]               0
            Conv1d-9              [-1, 128, 48]           24704
      BatchNorm1d-10              [-1, 128, 48]             256
             ReLU-11              [-1, 128, 48]               0
        MaxPool1d-12              [-1, 128, 24]               0
           Conv1d-13              [-1, 256, 24]           98560
      BatchNorm1d-14              [-1, 256, 24]             512
             ReLU-15              [-1, 256, 24]               0
        MaxPool1d-16              [-1, 256, 12]               0
           Conv1d-17              [-1, 512, 12]          393728
      BatchNorm1d-18              [-1, 512, 12]            1024
             ReLU-19              [-1, 512, 12]               0
        MaxPool1d-20               [-1, 512, 6]               0
AdaptiveAvgPool1d-21               [-1, 512, 1]               0
           Linear-22                  [-1, 512]          262656
             ReLU-23                  [-1, 512]               0
           Linear-24                    [-1, 4]            2052
================================================================
Total params: tensor(791972)
Trainable params: tensor(791972)
Non-trainable params: tensor(0)
