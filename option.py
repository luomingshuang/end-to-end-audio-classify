dataset = '/home/ceph/luomingshuang/LRW'
nClasses = 500
mode = 'temporalConv' #'backendGRU' or 'finetuneGRU'
every_frame = 'False'
lr = 0.0003
batch_size = 36
workers = 4
epochs = 30
interval = 10  #display interval
test = 'False'
savedir = '/home/yangshuang/luomingshuang/ASR/codes/end-to-end-lipreading/audio_only/weights'
#weights = ''