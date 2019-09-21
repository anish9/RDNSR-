"""config file --  
   ---default with learning rate annelaing
   ---create custom to turn annealer OFF"""

Total_epochs        =   40
Number_of_snapshots =   4 
Initial_LR          =   1e-3
RDB_DEPTH           =   6
upsample_dim        =   2
Train_count         =   463
Val_count           =   51
TENSORBOARD_DIR     =   "./tensorboard_files"


param_maps = {"epochs":Total_epochs,
              "snaps":Number_of_snapshots,
              "lr":Initial_LR,
              "Depth":RDB_DEPTH,
              "upsample_config":upsample_dim,
              "train_count":Train_count,
              "val_count":Val_count,
              "tb_writes":TENSORBOARD_DIR}