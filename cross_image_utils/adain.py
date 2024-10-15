import torch

def masked_adain_batch(content_feat, style_feat, content_mask, style_mask): # (3,4,64,64) (4,64,64) (64,64) -> (4,64,64)
    assert (content_feat.size()[-2:] == style_feat.size()[-2:])
    if len(content_mask.shape) == 2:
        # content_feat = content_feat.unsqueeze(0)
        # style_feat = style_feat.unsqueeze(0)
        content_mask = content_mask.unsqueeze(0)
        style_mask = style_mask.unsqueeze(0)
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_batch(style_feat, mask=style_mask)
    content_mean, content_std = calc_mean_std_batch(content_feat, mask=content_mask)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    style_normalized_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return content_feat * (1 - content_mask.unsqueeze(1)) + style_normalized_feat * content_mask.unsqueeze(1)
def masked_adain(content_feat, style_feat, content_mask, style_mask): # (4,64,64) (64,64) (64,64) -> (4,64,64)
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat, mask=style_mask)
    content_mean, content_std = calc_mean_std(content_feat, mask=content_mask)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    style_normalized_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return content_feat * (1 - content_mask) + style_normalized_feat * content_mask

def adain_batch(content_feat, style_feat):
    assert (content_feat.size()[-2:] == style_feat.size()[-2:]) # # prev: (4,64,64) cur: (chunk_size,64,64,3)
    if len(content_feat.shape) == 3:
        content_feat = content_feat.unsqueeze(0)
        style_feat = style_feat.unsqueeze(0)
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_batch(style_feat)
    content_mean, content_std = calc_mean_std_batch(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2]) # # prev: (4,64,64) cur: (chunk_size,64,64,3)
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5, mask=None):
    # 处理数据格式:(C,H,W)
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    if len(size) == 2:
        return calc_mean_std_2d(feat, eps, mask)

    assert (len(size) == 3)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps # (4,)
        feat_std = feat_var.sqrt().view(C, 1, 1) # (4,1,1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)

    return feat_mean, feat_std

def calc_mean_std_batch(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size() # (b,c,h,w)
    if len(size) == 3:
        return calc_mean_std(feat, eps, mask)

    assert (len(size) == 4)
    B,C = size[0],size[1]
    if mask is not None:
        feat_var_list = []
        feat_std_list = []
        feat_mean_list = []
        for i in range(B):
            feat_var = feat[i].view(C, -1)[:, mask[i].view(-1) == 1].var(dim=1) + eps
            feat_std = feat_var.sqrt().view(C, 1, 1)
            feat_mean = feat[i].view(C, -1)[:, mask[i].view(-1) == 1].mean(dim=1).view(C, 1, 1)
            feat_var_list.append(feat_var)
            feat_std_list.append(feat_std)
            feat_mean_list.append(feat_mean)
        feat_mean = torch.stack(feat_mean_list,dim=0)
        feat_std = torch.stack(feat_std_list,dim=0)
    else:
        feat_var = feat.view(B,C, -1).var(dim=-1) + eps
        feat_std = feat_var.sqrt().view(B,C, 1, 1)
        feat_mean = feat.view(B,C, -1).mean(dim=-1).view(B,C, 1, 1)

    return feat_mean, feat_std
def calc_mean_std_2d(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 2)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1)

    return feat_mean, feat_std


if __name__ == "__main__":
    # content_feat, style_feat, content_mask, style_mask = torch.zeros(3,4,64,64),torch.zeros(3,4,64,64),torch.zeros(3,64,64),torch.zeros(3,64,64)
    content_feat, style_feat, content_mask, style_mask = torch.zeros(4,64,64),torch.zeros(4,64,64),torch.zeros(64,64),torch.zeros(64,64)
    style_normalized_feat = masked_adain_batch(content_feat, style_feat, content_mask, style_mask)
    stylized = adain_batch(content_feat, style_feat)
    print(style_normalized_feat.shape,stylized.shape)