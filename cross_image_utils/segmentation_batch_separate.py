import os.path
from secrets import choice
from typing import Tuple, List
from utils import load_config, save_config
import nltk
import numpy as np
import torch
from sklearn.cluster import KMeans
import os
from constants import STYLE_INDEX, STRUCT_INDEX
import torch
from cross_image_utils.attention_utils import show_tensor_image,show_cluster_image
# 这两行download非常慢
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from cross_image_utils.attention_visualization import show_cross_attention_batch,show_self_attention_comp_batch
"""
Self-segmentation technique taken from Prompt Mixing: https://github.com/orpatashnik/local-prompt-mixing

processor当中首先update_attention

"""
from cross_image_utils.adain import masked_adain,adain


class Segmentor:
    def __init__(self,config ,tokenizer, res: int = 32):
        '''

        '''
        self.tokenizer = tokenizer
        self.prompt = config.prompt # a photo of a animal
        self.style_prompt = config.style_prompt
        self.num_segments = config.num_segments
        self.resolution = res
        self.object_nouns = [config.object_noun]
        self.style_object_nouns = [config.style_object_noun]
        # self.chunk_size = chunk_size
        self.chunk_flag = False
        tokenized_prompt = nltk.word_tokenize(self.prompt)
        forbidden_words = [word.upper() for word in ["photo", "image", "picture"]]
        self.nouns = [(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(tokenized_prompt)) 
                      if pos[:2] == 'NN' and word.upper() not in forbidden_words] # (i,word) 返回符合要求的索引位置和词本身
        # nltk.pos_tag：分词并且进行词性标注 [('A', 'DT'), ('photo', 'NN'), ('of', 'IN'), ('a', 'DT'), ('animal', 'NN')] NN表示名词
        style_tokenized_prompt = nltk.word_tokenize(self.style_prompt)
        self.style_nouns = [(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(style_tokenized_prompt))
                      if pos[:2] == 'NN' and word.upper() not in forbidden_words] # (i,word) 返回符合要求的索引位置和词本身
        # visualization
        self.config = config
        # self.cross_attention_dir = self.config.cross_attention_dir
        # self.self_attention_dir = self.config.self_attention_dir
        # self.cluster_path = self.config.cluster_path
        # self.mask_path = self.config.mask_path
    def setdirs(self,dirs):
        self.cross_attention_dir = dirs[0]
        self.self_attention_dir = dirs[1]
        self.cluster_path = dirs[2]
        self.mask_path = dirs[3]

    def update_attention(self, attn, is_cross):
        # 根据attn形状和类型（self or cross）更新内部属性
        res = int(attn.shape[2] ** 0.5) # attn_weight:(12,8,1024,1024)
        if attn.shape[0] > 3:
            self.chunk_flag = True
        if is_cross:
            if res == 16:
                self.cross_attention_32 = attn # (9,8,256,77) (batch,heads,size,token_ids)
            elif res == 32:
                self.cross_attention_64 = attn # (9,8,1024,77)
        else:
            b,heah,h,w = attn.shape
            if res == 32:
                if h == w:
                    self.self_attention_32 = attn
                else:
                    min_v = min(h,w)
                    self.self_attention_32 = attn[:,:,:min_v,:min_v]
            elif res == 64:
                if h == w:
                    self.self_attention_64 = attn # (4096,)
                else:
                    min_v = min(h,w)
                    self.self_attention_64 = attn[:,:,:min_v,:min_v]

    def __call__(self, *args, **kwargs):
        clusters = self.cluster_batch()
        cluster2noun = self.cluster2noun(clusters)
        return cluster2noun
    def cluster(self, res: int = 32):
        # 得到style和struct的聚类结果
        np.random.seed(1)
        self_attn = self.self_attention_32 if res == 32 else self.self_attention_64

        style_attn = self_attn[STYLE_INDEX].mean(dim=0).cpu().numpy()
        style_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(style_attn)
        style_clusters = style_kmeans.labels_.reshape(res, res) # (32,32)

        struct_attn = self_attn[STRUCT_INDEX].mean(dim=0).cpu().numpy()
        struct_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(struct_attn)
        struct_clusters = struct_kmeans.labels_.reshape(res, res)

        return style_clusters, struct_clusters
    def cluster_batch(self,choice="content", res: int = 32):
        # 得到style和struct的聚类结果
        np.random.seed(1)
        self_attn = self.self_attention_32 if res == 32 else self.self_attention_64
        if choice == "style":
            # 【3，8，1024，1024】--> [8,1024,1024] --> [1024,1024]
            style_attn = self_attn[self.chunk_size,:].mean(dim=0).cpu().numpy() # style只去一个进行计算[1024,1024]
            style_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(style_attn)
            style_clusters = style_kmeans.labels_.reshape(res, res)
            # 直接在第一个维度上扩展为 3，变成 (3, B, C, H, W)
            style_clusters_chunk = np.tile(style_clusters, (self.chunk_size, 1, 1)) # (3,32,32)
            return style_clusters_chunk
        if choice == "content":
            struct_clusters_list = []
            struct_attns = self_attn[:self.chunk_size].mean(dim=1).cpu().numpy()
            for struct_attn in struct_attns:
                struct_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(struct_attn)
                struct_cluster = struct_kmeans.labels_.reshape(res, res)
                struct_clusters_list.append(struct_cluster)
            struct_clusters = np.stack(struct_clusters_list)
            return struct_clusters

    def cluster2noun(self, clusters, cross_attn,attn_index):
        # 将聚类结果映射到名词上 clusters:(32,32) cross_attn:(8,256,77)
        result = {}
        res = int(cross_attn.shape[1] ** 0.5) # 256
        nouns_indices = [index for (index, word) in (self.nouns if (attn_index == STRUCT_INDEX) else self.style_nouns)]
        cross_attn = cross_attn.mean(dim=0).reshape(res, res, -1)  # (8,256,77) -> (256,77) -> (16,16,77)
        nouns_maps = cross_attn.cpu().numpy()[:, :, [i + 1 for i in nouns_indices]]  # (16,16,4)
        normalized_nouns_maps = np.zeros_like(nouns_maps).repeat(2, axis=0).repeat(2, axis=1)  # (32,32,4)
        for i in range(nouns_maps.shape[-1]):  # 名词个数
            curr_noun_map = nouns_maps[:, :, i].repeat(2, axis=0).repeat(2, axis=1)  # (32,32)
            normalized_nouns_maps[:, :, i] = (curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()

        max_score = 0
        all_scores = []
        for c in range(self.num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]  # （32，32）
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
            all_scores.append(max(scores))
            max_score = max(max(scores), max_score)

        all_scores.remove(max_score)  # list:5
        mean_score = sum(all_scores) / len(all_scores)

        for c in range(self.num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
            result[c] = (self.nouns if (attn_index == STRUCT_INDEX) else self.style_nouns)[np.argmax(np.array(scores))] if max(scores) > 1.4 * mean_score else "BG"
        # 记录对应表示前景还是背景
        return result  # {0: 'BG', 1: 'BG', 2: (1, 'tea'), 3: 'BG', 4: (1, 'tea')}

    # def cluster2noun_batch(self, clusters, cross_attn, attn_index):
    #     # 将聚类结果映射到名词上 clusters:(chunk_size,32,32) cross_attn:(12,8,256,77)
    #     result = {}
    #     res = int(cross_attn.shape[2] ** 0.5)
    #     nouns_indices = [index for (index, word) in self.nouns]
    #     cross_attn = cross_attn[attn_index*self.chunk_size:(attn_index+1)*self.chunk_size].mean(dim=1).reshape(self.chunk_size,res, res, -1)
    #     nouns_maps = cross_attn.cpu().numpy()[:,:, :, [i + 1 for i in nouns_indices]] # chunk,16,16,4
    #     normalized_nouns_maps = np.zeros_like(nouns_maps).repeat(2, axis=1).repeat(2, axis=2) # chunk,32,32,4
    #     for i in range(nouns_maps.shape[-1]):
    #         curr_noun_map = nouns_maps[:,:, :, i].repeat(2, axis=1).repeat(2, axis=2)
    #         normalized_nouns_maps[:,:, :, i] = (curr_noun_map - np.abs(np.min(curr_noun_map,axis=(1,2),keepdims=True))) / np.max(curr_noun_map,axis=(1,2),keepdims=True)
    #
    #     max_score = 0
    #     all_scores = []
    #     for c in range(self.num_segments):
    #         cluster_mask = np.zeros_like(clusters)
    #         cluster_mask[clusters == c] = 1
    #         score_maps = [cluster_mask * normalized_nouns_maps[:,:, :, i] for i in range(len(nouns_indices))]
    #         scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
    #         all_scores.append(max(scores))
    #         max_score = max(max(scores), max_score)
    #
    #     all_scores.remove(max_score)
    #     mean_score = sum(all_scores) / len(all_scores)
    #
    #     for c in range(self.num_segments):
    #         cluster_mask = np.zeros_like(clusters)
    #         cluster_mask[clusters == c] = 1
    #         score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
    #         scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
    #         result[c] = self.nouns[np.argmax(np.array(scores))] if max(scores) > 1.4 * mean_score else "BG"
    #
    #     return result

    def create_mask(self, clusters, cross_attention, attn_index,size=64):
        cluster2noun = self.cluster2noun(clusters, cross_attention[attn_index],attn_index)
        mask = clusters.copy() #
        obj_segments = [c for c in cluster2noun if cluster2noun[c][1] in (self.object_nouns if (attn_index == STRUCT_INDEX) else self.style_object_nouns)]
        for c in range(self.num_segments):
            mask[clusters == c] = 1 if c in obj_segments else 0
        mask = np.resize(mask, (size, size))
        return torch.from_numpy(mask).to("cuda")
    def create_mask_batch(self,clusters, cross_attention, attn_index,size=64):
        cur_cross_attentions = cross_attention[attn_index*self.chunk_size:(attn_index+1)*self.chunk_size] # (chunk,8,256,77)
        n = len(cur_cross_attentions)
        cur_batch_mask = []
        for i in range(n):
        # for cross_attention in cur_cross_attentions:
            cross_attention_i = cur_cross_attentions[i]
            cluster2noun = self.cluster2noun(clusters[i], cross_attention_i,attn_index)
            mask = clusters[i].copy()
            obj_segments = [c for c in cluster2noun if cluster2noun[c][1] in (self.object_nouns if (attn_index == STRUCT_INDEX) else self.style_object_nouns)]
            for c in range(self.num_segments):
                mask[clusters[i] == c] = 1 if c in obj_segments else 0
            mask = np.resize(mask, (size, size))
            cur_batch_mask.append(torch.from_numpy(mask).to("cuda"))
        return torch.stack(cur_batch_mask,dim=0)


    def set_style_mask(self,mask_style_32,mask_style_64):
        self.mask_style_32 = mask_style_32
        self.mask_style_64 = mask_style_64
    def get_object_masks(self,chunk_index) -> Tuple[torch.Tensor]:

        res = 32
        self.chunk_size = self.cross_attention_32.shape[0] // 3
        save_dir_cluster = str(self.cluster_path)
        if chunk_index == 0:
            # def show_cross_attention_batch(tokenizer,prompts,attention_maps,save_dir, select: int = 4):
            # latents[:self.segmentor.chunk_size]

            map_dict = {0:"stylized",1:"style",2:"struct"}
            for i in range(3):
                if i == 0 or i == 2:
                    prompt = self.prompt
                elif i == 1:
                    prompt = self.style_prompt
                else:
                    prompt = "A photo of an object"
                cross_attention_maps_32 = self.cross_attention_32[i*self.chunk_size:(i+1)*self.chunk_size,:].cpu() # tensor -> numpy (3,8,256,77)
                cur_save_dir = os.path.join(self.cross_attention_dir,map_dict[i])
                os.makedirs(cur_save_dir,exist_ok=True)
                show_cross_attention_batch(self.tokenizer,[prompt],cross_attention_maps_32,cur_save_dir,select=0) # select表示在list当中的索引
                cur_self_save_dir = os.path.join(self.self_attention_dir, map_dict[i])
                os.makedirs(cur_self_save_dir,exist_ok=True)
                self_attention_maps_32 = self.self_attention_32[i*self.chunk_size:(i+1)*self.chunk_size,:].cpu().numpy()
                show_self_attention_comp_batch(cur_self_save_dir,self_attention_maps_32)
                self_attention_maps_64 = self.self_attention_64[i*self.chunk_size:(i+1)*self.chunk_size,:].cpu().numpy()
                show_self_attention_comp_batch(cur_self_save_dir,self_attention_maps_64)
                # save_dir,attention_maps

        if self.chunk_flag:
            if chunk_index == 0:
                # clusters_struct_64 = self.cluster_batch(choice = "content",res=64)
                # clusters_style_64 = self.cluster_batch(choice = "style",res=64) # (3,32,32)
                # # style_mask可以直接在单个的基础上重复    struct必须每个单独分割
                # self.mask_style_64 = self.create_mask_batch(clusters_style_64, self.cross_attention_64, STYLE_INDEX)
                # self.mask_struct_64 = self.create_mask_batch(clusters_struct_64, self.cross_attention_64, STRUCT_INDEX)
                # show_tensor_image(str(self.mask_path),self.mask_style_64,'style',64)
                # show_tensor_image(str(self.mask_path),self.mask_struct_64, 'struct', 64)
                # # self.set_style_mask(mask_style_32,mask_style_64)

                clusters_struct = self.cluster_batch(choice = "content",res=res)
                clusters_style = self.cluster_batch(choice = "style",res=res) # (3,32,32)
                show_cluster_image(save_dir_cluster, clusters_struct, f'struct_{res}', segments=self.num_segments)
                show_cluster_image(save_dir_cluster, clusters_style, f'style_{res}', segments=self.num_segments)
                self.mask_style = self.create_mask_batch(clusters_style, self.cross_attention_32 if res == 32 else self.cross_attention_64, STYLE_INDEX)
                self.mask_struct = self.create_mask_batch(clusters_struct, self.cross_attention_32 if res == 32 else self.cross_attention_64, STRUCT_INDEX)
                show_tensor_image(str(self.mask_path), self.mask_style, 'style', str(res))
                show_tensor_image(str(self.mask_path), self.mask_struct, 'struct', str(res))

            else:
                clusters_struct = self.cluster_batch(choice="content", res=res)
                self.mask_struct = self.create_mask_batch(clusters_struct, self.cross_attention_32 if res == 32 else self.cross_attention_64, STRUCT_INDEX)

        else:
            clusters_style, clusters_struct = self.cluster(res=res)  # (64,64)
            if chunk_index == 0: # mask_style值保存一次
                self.mask_style = self.create_mask(clusters_style, self.cross_attention_32 if res == 32 else self.cross_attention_64, STYLE_INDEX)
            self.mask_struct = self.create_mask(clusters_struct, self.cross_attention_32 if res == 32 else self.cross_attention_64, STRUCT_INDEX)
            show_tensor_image(str(self.mask_path),self.mask_style,'style',str(res))
            show_tensor_image(str(self.mask_path),self.mask_struct, 'struct', str(res))
        return self.mask_style, self.mask_struct

if __name__ == "__main__":
    os.chdir("Codes/cross-image-attention")
    prompt = "a tea pot pouring tea into a cup."
    object_noun = "tea"
    chunk_size = 1
    config, cross_image_config = load_config()
    segmentor = Segmentor(config=config)
    attn_weight = torch.zeros((chunk_size*3,4,1024,1024))
    is_cross = False
    segmentor.update_attention(attn_weight, is_cross)
    attn_weight = torch.zeros((chunk_size*3,4,4096,4096))
    is_cross = False
    segmentor.update_attention(attn_weight, is_cross)
    attn_weight = torch.zeros((chunk_size*3,8,256,77))
    is_cross = True
    segmentor.update_attention(attn_weight, is_cross)
    attn_weight = torch.zeros((chunk_size*3,8,1024,77))
    is_cross = True
    segmentor.update_attention(attn_weight, is_cross)
    mask_style_32, mask_struct_32, mask_style_64, mask_struct_64 = segmentor.get_object_masks()
    print(mask_style_32.shape,mask_style_64.shape)

