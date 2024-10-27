import os.path
from typing import Tuple, List

import nltk
import numpy as np
import torch
from sklearn.cluster import KMeans
import os
from constants import STYLE_INDEX, STRUCT_INDEX
import torch
from cross_image_utils.attention_utils import show_tensor_image
# 这两行download非常慢
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

"""
Self-segmentation technique taken from Prompt Mixing: https://github.com/orpatashnik/local-prompt-mixing

processor当中首先update_attention
"""



class Segmentor:
    def __init__(self, prompt: str, object_nouns: List[str],chunk_size, num_segments: int = 5, res: int = 32):
        self.prompt = prompt # a photo of a animal
        self.num_segments = num_segments
        self.resolution = res
        self.object_nouns = object_nouns
        self.chunk_size = chunk_size
        self.chunk_flag = False
        tokenized_prompt = nltk.word_tokenize(prompt)
        forbidden_words = [word.upper() for word in ["photo", "image", "picture"]]
        self.nouns = [(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(tokenized_prompt)) 
                      if pos[:2] == 'NN' and word.upper() not in forbidden_words] # (i,word) 返回符合要求的索引位置和词本身
        # nltk.pos_tag：分词并且进行词性标注 [('A', 'DT'), ('photo', 'NN'), ('of', 'IN'), ('a', 'DT'), ('animal', 'NN')] NN表示名词

    def update_attention(self, attn, is_cross):
        # 根据attn形状和类型（self or cross）更新内部属性
        res = int(attn.shape[2] ** 0.5) # attn_weight:(12,8,1024,1024)
        if attn.shape[0] > 3:
            self.chunk_flag = True
        if is_cross:
            if res == 16:
                self.cross_attention_32 = attn
            elif res == 32:
                self.cross_attention_64 = attn
        else:
            if res == 32:
                self.self_attention_32 = attn
            elif res == 64:
                self.self_attention_64 = attn # (4096,)

    def __call__(self, *args, **kwargs):
        clusters = self.cluster_batch(res=32)
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
    def cluster_batch(self, res: int = 32):
        # 得到style和struct的聚类结果
        np.random.seed(1)
        self_attn = self.self_attention_32 if res == 32 else self.self_attention_64
        # 【3，8，1024，1024】--> [8,1024,1024] --> [1024,1024]
        style_attn = self_attn[self.chunk_size,:].mean(dim=0).cpu().numpy() # style只去一个进行计算[1024,1024]
        style_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(style_attn)
        style_clusters = style_kmeans.labels_.reshape(res, res)
        # 直接在第一个维度上扩展为 3，变成 (3, B, C, H, W)
        style_clusters_chunk = np.tile(style_clusters, (self.chunk_size, 1, 1)) # (3,32,32)

        struct_clusters_list = []
        struct_attns = self_attn[:self.chunk_size].mean(dim=1).cpu().numpy()
        for struct_attn in struct_attns:
            struct_kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(struct_attn)
            struct_cluster = struct_kmeans.labels_.reshape(res, res)
            struct_clusters_list.append(struct_cluster)
        struct_clusters = np.stack(struct_clusters_list)
        return style_clusters_chunk, struct_clusters

    def cluster2noun(self, clusters, cross_attn):
        # 将聚类结果映射到名词上 clusters:(32,32) cross_attn:(8,256,77)
        result = {}
        res = int(cross_attn.shape[1] ** 0.5) # 256
        nouns_indices = [index for (index, word) in self.nouns]
        cross_attn = cross_attn.mean(dim=0).reshape(res, res, -1)  # (256,77) -> (16,16,77)
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
            result[c] = self.nouns[np.argmax(np.array(scores))] if max(scores) > 1.4 * mean_score else "BG"
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

    def create_mask(self, clusters, cross_attention, attn_index):
        cluster2noun = self.cluster2noun(clusters, cross_attention[attn_index])
        mask = clusters.copy()
        obj_segments = [c for c in cluster2noun if cluster2noun[c][1] in self.object_nouns]
        for c in range(self.num_segments):
            mask[clusters == c] = 1 if c in obj_segments else 0
        return torch.from_numpy(mask).to("cuda")
    def create_mask_batch(self,clusters, cross_attention, attn_index):
        cur_cross_attentions = cross_attention[attn_index*self.chunk_size:(attn_index+1)*self.chunk_size] # (chunk,8,256,77)
        n = len(cur_cross_attentions)
        cur_batch_mask = []
        for i in range(n):
        # for cross_attention in cur_cross_attentions:
            cross_attention_i = cur_cross_attentions[i]
            cluster2noun = self.cluster2noun(clusters[i], cross_attention_i)
            mask = clusters[i].copy()
            obj_segments = [c for c in cluster2noun if cluster2noun[c][1] in self.object_nouns]
            for c in range(self.num_segments):
                mask[clusters[i] == c] = 1 if c in obj_segments else 0
            cur_batch_mask.append(torch.from_numpy(mask).to("cuda"))
        return torch.stack(cur_batch_mask,dim=0)



    def get_object_masks(self) -> Tuple[torch.Tensor]:
        self.chunk_size = self.cross_attention_32.shape[0] // 3
        if self.chunk_flag:

            clusters_style_32, clusters_struct_32 = self.cluster_batch(res=32) # (3,32,32)
            clusters_style_64, clusters_struct_64 = self.cluster_batch(res=64) # (3,32,32)

            # style_mask可以直接在单个的基础上重复    struct必须每个单独分割
            mask_style_32 = self.create_mask_batch(clusters_style_32, self.cross_attention_32, STYLE_INDEX)
            mask_style_64 = self.create_mask_batch(clusters_style_64, self.cross_attention_64, STYLE_INDEX)

            mask_struct_32 = self.create_mask_batch(clusters_struct_32, self.cross_attention_32, STRUCT_INDEX)
            mask_struct_64 = self.create_mask_batch(clusters_struct_64, self.cross_attention_64, STRUCT_INDEX)

            save_dir = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/debug/masks"
            show_tensor_image(save_dir,mask_style_64,'style',64)
            show_tensor_image(save_dir,mask_struct_64, 'struct', 64)
        else:

            clusters_style_32, clusters_struct_32 = self.cluster(res=32)  # (32,32)
            clusters_style_64, clusters_struct_64 = self.cluster(res=64)  # (64,64)
            # self.cross_attention_32:(3,8,256,77)
            mask_style_32 = self.create_mask(clusters_style_32, self.cross_attention_32, STYLE_INDEX)  # （32，32）
            mask_struct_32 = self.create_mask(clusters_struct_32, self.cross_attention_32, STRUCT_INDEX)
            mask_style_64 = self.create_mask(clusters_style_64, self.cross_attention_64, STYLE_INDEX)
            mask_struct_64 = self.create_mask(clusters_struct_64, self.cross_attention_64, STRUCT_INDEX)
            save_dir = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/debug/masks"
            show_tensor_image(save_dir,mask_style_64,'style',64)
            show_tensor_image(save_dir,mask_struct_64, 'struct', 64)
        return mask_style_32, mask_struct_32, mask_style_64, mask_struct_64

if __name__ == "__main__":
    prompt = "a tea pot pouring tea into a cup."
    object_noun = "tea"
    chunk_size = 1
    segmentor = Segmentor(prompt=prompt, object_nouns=[object_noun], chunk_size=chunk_size)
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
