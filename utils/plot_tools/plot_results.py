# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/12 14:41:17
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 预测结果可视化
=================================================
'''
import os 
from tqdm import tqdm
from skimage.util import montage
import nibabel as nib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')  # 设置非交互式后端
        

class NiiViewer:
    def __init__(self, nii_dir, modal='t1', cmap='bone', outputs_dir='./outputs', fontsize=8, font_color='red') -> None:
        self.nii_dir = nii_dir
        self.modal = modal
        self.cmap = cmap
        self.outputs_dir = outputs_dir
        self.fontsize = fontsize
        self.font_color = font_color
        self._data_dict = self._load_data_dict()  # 用于存储数据字典的实例变量

        assert modal in self._data_dict, f"Modal {modal} not found in data."
        assert 'mask' in self._data_dict, "Mask data not found."
        assert 'pred' in self._data_dict, "Prediction data not found."

        self.img = self._data_dict[modal]
        self.mask = self._data_dict['mask']
        self.pred = self._data_dict['pred']

        self.slices = []

        for slice_n in tqdm(range(self.img.shape[-1])):
            self.slices.append(self._prefetch_slices(slice_n))
    
    def _prefetch_slices(self, slice_n):
        slices = {}
        for axis in ['x', 'y', 'z']:
            slices[axis] ={
                'img': np.rot90(self._get_slice(self.img, axis, slice_n), k=-1),
                'mask': np.rot90(self._get_slice(self.mask, axis, slice_n), k=-1),
                'pred': np.rot90(self._get_slice(self.pred, axis, slice_n), k=-1)
            }
        return slices
    
    def _load_data_dict(self) -> dict:
        """加载并返回数据字典。"""
        data_dict = {}
        file_list = self._get_file_list(self.nii_dir)
        for file_path in file_list:
            data_modal = self._get_data_modal(file_path)
            data_dict[data_modal] = self._get_nii_data(file_path)
        return data_dict
    
    def show_one_slice(self, slice_n):
        """可视化一个切片。"""
        
        modalities = [self.modal, 'mask', 'pred']
        titles = ['Image', 'Label', 'Prediction']
        
        fig, ax = plt.subplots(3, 3, figsize=(15, 5))

        for i, mod in enumerate(modalities):
            data = self._data_dict[mod]
            for j, axis in enumerate(['x', 'y', 'z']):
                slice_data = np.rot90(self._get_slice(data, axis, slice_n), k=-1)
                ax[i, j].imshow(slice_data, cmap=self.cmap)

                ax[i, j].set_title(f"{titles[i]} slice {slice_n} along the {axis}-axis", 
                                  fontsize=self.fontsize, color=self.font_color)
                ax[i, j].axis('off')
        
        fig.tight_layout()
        if self.save_fig:
            self.save_fig(fig, self.outputs_dir, f'{self.modal}_slice_{slice_n}.png')
        plt.close()

    
    def show_color_map(self, slice_n):


        """可视化彩色图像。"""
        slices = self.slices[slice_n]

        img_x, img_y, img_z = slices['x']['img'], slices['y']['img'], slices['z']['img']
        mask_x, mask_y, mask_z = slices['x']['mask'], slices['y']['mask'], slices['z']['mask']
        pred_x, pred_y, pred_z = slices['x']['pred'], slices['y']['pred'], slices['z']['pred']

        fig, ax = plt.subplots(3, 2, figsize=(20, 20))
        plt.title(f'{self.modal}_2d_{slice_n}')
        ax[0, 0].imshow(img_x, cmap='bone')
        ax[0, 0].imshow(np.ma.masked_where(mask_x == False, mask_x), cmap='cool', alpha=0.6, animated=True)
        ax[0, 0].set_title('GT_x')

        ax[0, 1].imshow(img_x, cmap='bone')
        ax[0, 1].imshow(np.ma.masked_where(pred_x == False, pred_x), cmap='cool', alpha=0.6, animated=True)
        ax[0, 1].set_title('Prediction_x')


        ax[1, 0].imshow(img_y, cmap='bone')
        ax[1, 0].imshow(np.ma.masked_where(mask_y == False, mask_y), cmap='cool', alpha=0.6, animated=True)
        ax[1, 0].set_title('GT_y')

        ax[1, 1].imshow(img_y, cmap='bone')
        ax[1, 1].imshow(np.ma.masked_where(pred_y == False, pred_y), cmap='cool', alpha=0.6, animated=True)
        ax[1, 1].set_title('Prediction_y')

        ax[2, 0].imshow(img_z, cmap='bone')
        ax[2, 0].imshow(np.ma.masked_where(mask_z == False, mask_z), cmap='cool', alpha=0.6, animated=True)
        ax[2, 0].set_title('GT_z')

        ax[2, 1].imshow(img_z, cmap='bone')
        ax[2, 1].imshow(np.ma.masked_where(pred_z == False, pred_z), cmap='cool', alpha=0.6, animated=True)
        ax[2, 1].set_title('Prediction_z')

        for i in ax:
            for j in i:
                j.axis('off')

        fig.tight_layout()
        if self.save_fig:
            self.save_fig(fig, self.outputs_dir, f'{self.modal}_colormap_{slice_n}.png')
        plt.close()
        

    def show_all_slices(self, cmap='cool', alpha=0.6, animated=True):
        for slice_n in tqdm(range(len(self.slices))):
            self.show_color_map(slice_n)
        
    def show_montage(self, cmap='cool', alpha=0.6, animated=True):
        img = np.rot90(montage(self.img), k=3)
        mask = np.rot90(montage(self.mask), k=3)
        pred = np.rot90(montage(self.pred), k=3)

        fig, ax1 = plt.subplots(2, 1, figsize=(40, 20))
        plt.title(f'Montage of all slices of {self.modal}')

        ax1[0].imshow(img, cmap='bone')
        ax1[0].imshow(np.ma.masked_where(mask == False, mask), cmap=cmap, alpha=alpha, animated=animated)
        ax1[0].set_title('GT')

        ax1[1].imshow(img, cmap='bone')
        ax1[1].imshow(np.ma.masked_where(pred == False, pred), cmap=cmap, alpha=alpha, animated=animated)
        ax1[1].set_title('Prediction')

        for i in ax1:
            i.axis('off')

        if self.save_fig:
            self.save_fig(fig, self.outputs_dir, f'{self.modal}_montage_3d_to_2d.png')

        plt.close()


    def save_fig(self, fig, dir_path, filename, dpi=600, format='png', bbox_inches='tight'):
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, filename)
        fig.savefig(file_path, dpi=dpi, format=format, bbox_inches=bbox_inches)
        
    
    def _get_slice(self, data, axis, slice_n):
        """根据给定的轴向和切片索引获取切片数据。"""
        if axis == 'x':
            return data[slice_n, :, :]
        elif axis == 'y':
            return data[:, slice_n, :]
        elif axis == 'z':
            return data[:, :, slice_n]
        else:
            raise ValueError(f"Invalid axis: {axis}")
        
    def _get_file_list(self, nii_dir):
        """
        获取目录下所有文件的路径列表
        """
        file_list = os.listdir(nii_dir)
        file_list.sort()
        file_list = [os.path.join(nii_dir, file) for file in file_list]
        return file_list

    def _get_data_modal(self, file_path):
        return file_path.replace('/', '.').split('.')[-3].split('_')[-1]
    
    def _get_nii_data(self, nii_path):
        nii_data = nib.load(nii_path).get_fdata()
        return nii_data
    
if __name__ == "__main__":
    nii_dir = "/root/workspace/Helium-327-SegBrats/outputs/UNet3D_BN_best_ckpt@epoch8_diceloss0.2315_dice0.8134_7/P0"
    outputs_path = '/root/workspace/Helium-327-SegBrats/outputs'

    nii_viewer = NiiViewer(nii_dir, cmap='bone')
    data_dict = nii_viewer._data_dict

    # print(data_dict.keys())
    # print(data_dict['t1'].shape)

    # nii_viewer.show_one_slice(80, 'flair')
    # nii_viewer.show_color_map(80)
    # nii_viewer.show_montage()
    img = nii_viewer.img
    print(img.shape)
    nii_viewer.show_all_slices()