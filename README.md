# DeepVol

Volume inference for an input fruit image writen in caffe.
Run volumeFood.py, you may need to add your own caffe path





The dataset can be downloaded on 

The structures are like:
    data0_230

        video0
        
            0.jpg
            
            10.jpg
            ...
            310.jpg
        video1
        ...
        video9
    data1_300
    ...
    data27_160

Each folder include ONE fruit (e.g. data0_239) under DIFFERENT scenes (e.g. video0).
Format: data0_230, dataset index is 0, volume is 230 cm^3
