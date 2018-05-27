import numpy as np
import tensorflow as tf

from trainer import *
from tester import *
from trainer_256 import *
from config import get_config
from utils import prepare_dirs_and_logger, save_config

import pdb, os

def main(config):
    prepare_dirs_and_logger(config)

    if config.gpu>-1:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu)
    config.data_format = 'NHWC'



    if 1==config.model: ## 50
        trainer = DPIG_Encoder_GAN_BodyROI_FgBg(config)
        trainer.init_net()    
    if 2==config.model: ## 27
        trainer = DPIG_PoseRCV_AE_BodyROI(config)
        trainer.init_net()

    if 3==config.model: ## 51
        trainer = DPIG_Encoder_subSampleAppNetFgBg_GAN_BodyROI(config)
        trainer.init_net()
    if 4==config.model: ## 28
        trainer = DPIG_subnetSamplePoseRCV_GAN_BodyROI(config)
        trainer.init_net()




    if 11==config.model: ## 129
        trainer = DPIG_FourNetsFgBg_testOnly(config)
        trainer.init_net()
    if 12==config.model: ## 131
        trainer = DPIG_FourNetsFgBg_testOnlyCondition(config)
        trainer.init_net()
    if 13==config.model: ## 137
        trainer = DPIG_FourNetsFgBg_testOnlySampleFactor(config)
        trainer.init_net()



    ######################## trainer_256 ######################
    if 101==config.model:  ## 10017
        trainer = DPIG_Encoder_GAN_BodyROI_256(config)
        trainer.init_net()
    if 102==config.model:  ## 10027
        trainer = DPIG_PoseRCV_AE_BodyROI_256(config)
        trainer.init_net()

    if 103==config.model:  ## 10020
        trainer = DPIG_Encoder_subSampleAppNet_GAN_BodyROI_256(config)
        trainer.init_net()
    if 104==config.model:  ## 10028
        trainer = DPIG_subnetSamplePoseRCV_GAN_BodyROI_256(config)
        trainer.init_net()


    if 1001==config.model:  ## 100131
        trainer = DPIG_ThreeNetsApp_testOnlyCondition_256(config)
        trainer.init_net()
    if 1002==config.model:  ## 100137
        trainer = DPIG_ThreeNetsApp_testOnlySampleFactor_256(config)
        trainer.init_net()

    # if 1003==config.model:  ## 100131
    #     trainer = DPIG_ThreeNetsApp_testOnlyCondition_Vis_256(config)
    #     trainer.init_net()



    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        # if not config.load_path:
        #     raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
