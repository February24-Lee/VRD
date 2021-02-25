import torch, torchvision
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision.transforms import ToTensor

import numpy as np

from PIL import Image
from typing import Tuple
import json

# --- there are no information about img.
EXCEPT_IMG_TEST = [
    '9055930159_9560984041_b.jpg' , '5866509996_1c95a5c376_b.jpg' , '3988529173_9c88fa79cd_b.jpg' , 
    '4607898043_b7e3262887_b.jpg' , '8308021888_862a05c775_b.jpg' , '2393973731_1fe2908280_o.jpg' , 
    '4679659935_2cf8905ec5_b.jpg' , '5471365300_ac5b8723bf_b.jpg' , '4979927771_6d32994354_b.jpg' , 
    '319629473_aab7f52dc6_b.jpg' , '4205648420_a78d955dec_o.jpg' , '500336501_2ab8bd519b_b.jpg' , 
    '35355235_8955a0569b_o.jpg' , '9901558526_1fdcd0227c_b.jpg' , '145671186_e44c79795e_o.jpg' , 
    '8169707409_06eaa858ed_b.jpg' , '362154704_40edaae16c_b.jpg' , '6872898747_c7b540e84c_b.jpg' , 
    '4992656841_e8f3f51653_b.jpg' , '9704566676_770a125c4b_b.jpg' , '7975438142_3cbd1ac2f0_b.jpg' , 
    '9515859651_05c67290f0_b.jpg' , '1497493960_38538d32e7_b.jpg' , '5695527828_50d0b012cb_b.jpg' , 
    '2421762255_e3251311a2_b.jpg' , '7254840318_62657d3bbe_b.jpg' , '2304707627_0e3931b372_b.jpg' , 
    '5000000882_f65fbf6ffa_b.jpg' , '8968341898_7bc6d01265_b.jpg' , '4668514133_66126b2ce2_b.jpg' , 
    '3882370446_37f2d5cbbf_b.jpg' , '8315461526_a5f45166f1_b.jpg' , '9434661655_c3e6b4e8f6_b.jpg' , 
    '9276370952_60b0d448e2_b.jpg' , '8453972903_f93f5449dc_b.jpg' , '6766399563_3348857c51_b.jpg' , 
    '3903095433_6a62ea473b_b.jpg' , '9590214137_e68966ca93_b.jpg' , '3386170394_c117fa9f15_o.jpg' , 
    '2484981229_5c8f09bc9d_b.jpg' , '253724899_22356683f5_b.jpg' , '2084710251_92574f8f73_b.jpg' , 
    '5046517526_eaafe2ffb6_b.jpg' , '6343984716_425239576b_b.jpg' , '9492217660_c917418963_b.jpg'
]
EXCEPT_IMG_TRAIN=[
    '54717484_77b35c24b7_b.jpg' , '5964114151_282a76b84b_b.jpg' , '427087053_84fbf5422a_o.jpg' , 
    '5352850949_699f361c8f_b.jpg' , '500949431_a86557d8d6_b.jpg' , '5277318135_1f95e78c31_b.jpg' , 
    '7342787664_35c99f8923_b.jpg' , '2997556599_6baa5b71b6_b.jpg' , '4358562117_3348f800f8_b.jpg' , 
    '8735436371_a967e3a4ca_b.jpg' , '9544408990_0751a672ab_b.jpg' , '3555567846_bb0c84b2c1_b.jpg' , 
    '8610852982_9511cb455e_b.jpg' , '5202666001_b668afbfaa_b.jpg' , '4953831714_63d158cc58_b.jpg' , 
    '3746071295_79515b4fce_b.jpg' , '8057690200_121b2b38e4_b.jpg' , '3143628615_183fc1ec98_o.jpg' , 
    '8155274991_a99bfe173a_b.jpg' , '4198830374_8327492928_b.jpg' , '402154584_c07c1b2451_o.jpg' , 
    '4231258586_cff9da4b47_b.jpg' , '3683085307_e46cf54642_o.jpg' , '3187244759_a76991e995_b.jpg' , 
    '5330495089_a05ec9a970_b.jpg' , '6847412620_1d5316899d_b.jpg' , '467226200_355dd89e15_o.jpg' , 
    '8309375042_e0904087d0_b.jpg' , '8274563909_1a2ebb542d_o.jpg' , '8544815951_de514c0305_b.jpg' , 
    '5694191645_b7a1694e91_b.jpg' , '7191502136_07814ae258_b.jpg' , '9515748684_69eb6e9c36_b.jpg' , 
    '6745267175_e8c755ed9a_b.jpg' , '8661386437_3e9fea52a1_b.jpg' , '1279041211_1dbc1e1473_b.jpg' , 
    '5821765347_9426556f5e_b.jpg' , '2620141196_10f0a4e41b_o.jpg' , '4059299826_3f0f3710e3_b.jpg' , 
    '9677717021_9149b7e10e_b.jpg' , '8755166252_dd5df3cc79_b.jpg' , '2570945244_9c18ce1689_o.jpg' , 
    '4927799379_ec3183ae03_b.jpg' , '8493712073_f22d709dbc_b.jpg' , '7421419632_a80a690fec_b.jpg' , 
    '154454735_9c26394542_b.jpg' , '3311700989_25bfd12104_o.jpg' , '9461640166_192dbb5562_b.jpg' , 
    '8315131764_3a25a527f7_b.jpg' , '6777996934_0dc6518569_b.jpg' , '8706013136_6e865703a0_b.jpg' , 
    '6346999819_077411d5ea_b.jpg' , '5101929759_5dacb1b1cd_b.jpg' , '592250385_bfa81dac6b_o.jpg' , 
    '176688288_a4d7baca75_b.jpg' , '226695257_95810e49e2_b.jpg' , '8832771966_6ff658ed2d_b.jpg' , 
    '4464720916_332d36b5de_b.jpg' , '4528942592_0b54636ece_b.jpg' , '6946405750_4e6f727352_o.jpg' , 
    '9535192327_66cd4ecc0c_b.jpg' , '7967347560_c74c6bc261_b.jpg' , '80390700_5b76320e21_o.jpg' , 
    '4119885956_d12b94f031_b.jpg' , '9738925753_51b7b0a482_b.jpg' , '2976961387_0b94a88e7a_b.jpg' , 
    '4412116817_78ca013edc_o.jpg' , '4899027690_575a9b68e2_b.jpg' , '7572877642_a6bde87e5c_b.jpg' , 
    '3462144862_a6ed3f5561_b.jpg' , '2035630911_4e362eee89_b.jpg' , '97000333_9c6b968a17_o.jpg' , 
    '5434071453_5eded254be_b.jpg' , '9592354594_f4119c9f5b_o.jpg' , '3812454603_06a6b8fe3a_o.jpg' , 
    '5055217943_1f4ac4c71a_b.jpg' , '7694584902_0b420966b2_b.jpg' , '9109078971_37f36d3c31_b.jpg' , 
    '8897548122_66c2188e06_b.jpg' , '9388660821_9fd461b0a0_b.jpg' , '1231559434_bf8f286de9_b.jpg' , 
    '2435662132_bdbe52a942_b.jpg' , '4380431144_7fbb636bd8_o.jpg' , '9636666194_484da07806_b.jpg' , 
    '8421663346_793e1f2611_b.jpg' , '3966006696_60f1697888_b.jpg' , '6928176644_30e0d46e83_b.jpg' , 
    '5189191742_3b58f577ef_b.jpg' , '3628833918_70343358c4_b.jpg' , '7684877868_8bd564a13c_b.jpg' , 
    '6888806901_8cb74c8164_b.jpg' , '8085946335_372d0e07fb_b.jpg' , '6351971567_a0b5a7b8c5_b.jpg' , 
    '2949376249_974023302f_b.jpg' , '412008642_d90fb491c6_o.jpg' , '528114146_97e509f887_b.jpg' , 
    '8944478397_bdbefa9dd3_b.jpg' , '3826576380_aff0c8d9cb_b.jpg' , '10211332813_44f6c2c260_b.jpg' , 
    '5795255410_163f05a0b2_b.jpg' , '8517300599_8d91a34831_b.jpg' , '4414979299_7e82616ece_b.jpg' , 
    '5754303979_ca64a9ed4e_b.jpg' , '3129895560_c8132b20ed_o.jpg' , '10174682414_a2cb2ef93b_b.jpg' , 
    '8675511246_5190ed875d_b.jpg' , '4659876493_fdb7eacc51_b.jpg' , '6208863729_5c76f9817d_b.jpg' , 
    '8472477677_abca9d6de2_b.jpg' , '1139790441_3677ec2475_b.jpg' , '4183709664_62876db93f_b.jpg' , 
    '386029114_b1b7bd65fa_b.jpg' , '3675959757_48e8b715a7_o.jpg' , '320515692_1d31d084cb_b.jpg' , 
    '5895085186_5577e3cb28_b.jpg' , '8609741694_76c9381a76_b.jpg' , '5087396277_78c1b4e90b_b.jpg' , 
    '4397238687_7381227faf_b.jpg' , '277563055_33a9e5d0fc_o.jpg' , '5284662497_ee10b617f8_o.jpg' , 
    '8706792663_7685392ffa_b.jpg' , '9216314821_1e054cfdbd_b.jpg' , '4062790693_faa6b6ec02_o.jpg' , 
    '8261319540_e17bcfc378_b.jpg' , '4028639246_a39a4e6a80_o.jpg' , '8586023764_6665f5dc54_b.jpg' , 
    '5421727357_a11ba60fd3_b.jpg' , '7965295478_c80e1f182e_b.jpg' , '2382751079_a104b112ba_o.jpg' , 
    '1670237158_417a49ff80_b.jpg' , '3441289298_f80ce3a5dc_b.jpg' , '2743927562_2308175c25_b.jpg' , 
    '2637009044_7bd44f2924_b.jpg' , '9113558075_127523102d_b.jpg' , '5141397019_5090e77754_b.jpg' , 
    '2531859482_e4cef9049e_b.jpg' , '5865653056_50cb9760ec_b.jpg' , '4235760948_97545be40c_b.jpg' , 
    '2478472358_f0f787361e_b.jpg' , '4877920087_8b67de54af_o.jpg' , '8176830330_6a902e20b3_b.jpg' , 
    '8929868816_1bfb1d65d9_b.jpg' , '8637891557_c31e31ebe4_o.jpg' , '8890860889_685ee7ca3b_b.jpg' , 
    '2311317124_7ec72df548_o.jpg' , '7861450930_f2d873f19e_b.jpg' , '2313586264_d679193141_b.jpg' , 
    '2466990924_1655ce6e98_b.jpg' , '2736665098_0b0870f51f_b.jpg' , '7411589196_3af4c4a4e7_b.jpg' , '3972563291_87f0659dbd_b.jpg' , '5607314225_613a6fc8ac_b.jpg' , '8043341901_94509d0f43_b.jpg' , '10131940315_e8b4e6602e_b.jpg' , '8611606589_4f19153fe2_b.jpg' , '1749756225_604d684926_b.jpg' , '2446861169_85e9f50e30_b.jpg' , '3143515596_aa9365cb9e_o.jpg' , '2467357710_37838b3122_b.jpg' , '366978953_bf27d1b9a7_b.jpg' , '2307535945_515b5f0f23_b.jpg' , '3414330453_bfde388314_b.jpg' , '9174164813_c4d6c2751d_b.jpg' , '3072825572_2a0c8e4f51_b.jpg' , '118462037_f546324a73_o.jpg' , '5738994224_fa57777214_o.jpg' , '7656161610_600d5a26bc_o.jpg' , '6904362493_4a5b1aa80e_b.jpg' , '3702979469_86259e9f6d_b.jpg' , '3422389948_6dd9bf9f4e_b.jpg' , '8694884088_806b9e32b0_b.jpg' , '7872056026_2efdcc5d0f_b.jpg' , '3999356112_d5c099c4a9_b.jpg' , '2305755863_89e01bbb11_b.jpg' , '7915559940_4e2e2b061d_b.jpg' , '2390878558_257b6e24ae_b.jpg' , '4480001146_a09055c4d3_o.jpg' , '8503750156_019f7279d3_b.jpg' , '3178453157_6f966c3874_b.jpg' , '3696856176_c5fea96341_b.jpg' , '6717778167_f34980cfc1_b.jpg' , '8422945140_853bef5e46_b.jpg' , '1678250983_3d3dacc013_b.jpg' , '8236205786_27d8954af8_b.jpg' , '9297492040_f7d4cf428c_b.jpg' , '7166643364_b0efc5e6de_b.jpg' , '4276633939_9fcf93ce62_b.jpg' , '6864145983_62896c1028_b.jpg' , '4654946936_7e35c45776_b.jpg' , '3696491968_614601564a_o.jpg' , '2828905413_30fe7ef5d8_o.jpg' , '4957436828_9eb4368fc2_b.jpg' , '3072851331_2263467552_o.jpg' , '4015537255_ba5cac459a_b.jpg' , '437294626_72b563959e_o.jpg' , '3143560962_4b6f6756a3_b.jpg' , '3472426520_bc3b95cdd8_b.jpg' , '2893695320_90a0ba83d1_o.jpg' , '8658352018_ca66d66507_b.jpg' , '4949070716_08012e13e7_b.jpg' , '7868611572_837affb3ac_b.jpg' , '8631153694_44e5f5e8fd_b.jpg' , '6947188249_d14cca6b98_b.jpg' , '5762100665_5cf16d2721_b.jpg' , '2367899908_8e0bd3fd3b_b.jpg' , '1043510993_971f1e5626_b.jpg' , '9496685848_3cde6618fb_b.jpg' , '3196312539_a8fdeb5a98_b.jpg' , '2163968511_6b44aa0fb2_b.jpg' , '9544061327_f675900fa2_b.jpg' , '8731445489_af90aa948d_b.jpg' , '8100812502_c4b5bb0693_b.jpg' , '8724068884_c01e5740f1_b.jpg' , '9000038599_a2ca739f3e_b.jpg' , '4222842955_a8c2d07bb6_o.jpg' , '2340792779_b75c9d7803_b.jpg' , '8335963303_14fe847038_b.jpg' , '6905163310_e4901e0069_b.jpg' , '9138847028_8058e25439_b.jpg' , '9334971036_8f8bfbe73d_b.jpg'
]

class VRD_DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_vrd_object_list_path:str,
                 train_vrd_annotations_json_path:str,
                 train_sg_annotations_json_path:str,
                 train_img_folder_path:str,
                 test_vrd_object_list_path:str,
                 test_vrd_annotations_json_path:str,
                 test_sg_annotations_json_path:str,
                 test_img_folder_path:str,
                 test_shuffle = True,
                 test_drop_last = True,
		 train_shuffle= True,
		 train_drop_last=True):
        super(VRD_DataModule, self).__init__()
        self.train_ds = VRD_Dataset(train_vrd_object_list_path,
                                        train_vrd_annotations_json_path,
                                        train_sg_annotations_json_path,
                                        train_img_folder_path)
        self.test_ds = VRD_Dataset(test_vrd_object_list_path,
                                        test_vrd_annotations_json_path,
                                        test_sg_annotations_json_path,
                                        test_img_folder_path)
        self.test_shuffle =test_shuffle
        self.train_shuffle =train_shuffle
        self.test_drop_last = test_drop_last
        self.train_drop_last = train_drop_last
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          shuffle=self.train_shuffle,
                          drop_last=self.train_drop_last)
    
    def val_dataloader(self):
        return DataLoader(self.test_ds, 
                          shuffle=self.test_shuffle,
                          drop_last=self.test_drop_last)


class VRD_Dataset(Dataset):
    def __init__(self,
                 vrd_object_list_path:str,
                 vrd_annotations_json_path:str,
                 sg_annotations_json_path:str,
                 img_folder_path:str):
        '''
        '''
        super(VRD_Dataset, self).__init__()
        try:
            with open(vrd_object_list_path) as f:
                self.object_list = json.load(f)    
        except:
            print('can\'t open the object label file')
        
        try:
            with open(vrd_annotations_json_path) as f:
                self.vrd_annotations = json.load(f)    
        except:
            print('can\'t open the vrd_annotations_json_path')
        
        try:
            with open(sg_annotations_json_path) as f:
                self.sg_annotations_json = json.load(f)    
        except:
            print('can\'t open the sg_annotations_json_path')    
        
        if img_folder_path[-1] != '/':
            img_folder_path = img_folder_path + '/'
        self.img_folder_path = img_folder_path
        self.img_name_list = [key for key, value in self.vrd_annotations.items() if len(value) !=0 ]
        #img_name_list = list(self.vrd_annotations.keys())
    
    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self, idx) -> Tuple[Image.Image, float, float, torch.Tensor, torch.Tensor]:
        img_file_name = self.img_name_list[idx]
        
        # --- Load img
        image = Image.open(self.img_folder_path + img_file_name)
        
        # --- img infomation
        find_img_info = lambda value, f_list : filter(lambda x : x['filename']==value, f_list)
        img_info = list(find_img_info(img_file_name, self.sg_annotations_json))[0]
        img_w, img_h = img_info['width'], img_info['height']
        
        # --- class and bb
        # bb format YMIN, YMAX, XMIN, XMAX
        # it shuld be change at XMIN, YMIN, XMAX, YMAX
        img_anno = self.vrd_annotations[img_file_name]
        cls_bb_list = []
        for object_idx in img_anno:
            object_info = list(object_idx['object'].values())
            object_info = [object_info[0], object_info[1][2], object_info[1][0], object_info[1][3], object_info[1][1]]
            subject_info = list(object_idx['subject'].values())
            subject_info = [subject_info[0], subject_info[1][2], subject_info[1][0], subject_info[1][3], subject_info[1][1]]
            if object_info not in cls_bb_list:
                cls_bb_list.append(object_info)
            
            if subject_info not in cls_bb_list:
                cls_bb_list.append(subject_info)
        cls_bb_list = np.array(cls_bb_list)
        return (ToTensor()(image), 
                img_w, 
                img_h, 
                torch.tensor(cls_bb_list[:,0], dtype=torch.int64), 
                torch.tensor(cls_bb_list[:,1:], dtype=torch.float32))
                
        
    
