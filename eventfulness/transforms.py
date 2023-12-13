import torch
import torch.nn as nn


class ConvertBHWCtoBCHW(nn.Module):
    """Convert tensor from (B, H, W, C) to (B, C, H, W)
    """

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)


class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)
    """

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class CropAndRandomShuffle(object):
    '''
    Subdivide tensor in spatial dimensions and shuffle the subdivided blocks.
    '''
    def __init__(self, crop_num):
        assert isinstance(crop_num, (int, tuple))
        self.crop_num = crop_num
        self.crop_H_n = crop_num[0]
        self.crop_W_n = crop_num[1]

    @staticmethod
    def oneDidxTo2Didx(i, H):
        return (i // H, i % H)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        h = sample.size(2)
        w = sample.size(3)
        # crop_Cn = sample.size(5)

        crop_H_size = h // self.crop_H_n
        crop_W_size = w // self.crop_W_n

        crop_H_margin = h - crop_H_size * self.crop_H_n
        crop_W_margin = w - crop_W_size * self.crop_W_n

        crop_H_left_margin = crop_H_margin // 2
        crop_W_left_margin = crop_W_margin // 2

        crop_H_right_margin = crop_H_margin - crop_H_left_margin
        crop_W_right_margin = crop_W_margin - crop_W_left_margin

        total_crop_blocks_n = self.crop_H_n * self.crop_W_n
        shuffled_idx = torch.randperm(total_crop_blocks_n)

        old_idxes = torch.as_strided(torch.arange(total_crop_blocks_n), (self.crop_H_n, self.crop_W_n), (self.crop_W_n, 1))
        new_idxes = torch.as_strided(shuffled_idx, (self.crop_H_n, self.crop_W_n), (self.crop_W_n, 1))
        # print(f"crop h size {crop_H_size} crop w size {crop_W_size}")
        # print(f"margin {crop_H_margin} {crop_W_margin}")
        # print(f"left margin {crop_H_left_margin} {crop_W_left_margin}")
        shuffled_sample = torch.zeros((sample.size(0), sample.size(1),
                                       sample.size(2) - crop_H_margin, sample.size(3) - crop_W_margin))
        for r in range(old_idxes.size(0)):
            for c in range(old_idxes.size(1)):
                old_idx = CropAndRandomShuffle.oneDidxTo2Didx(old_idxes[r,c], self.crop_H_n)
                new_idx = CropAndRandomShuffle.oneDidxTo2Didx(new_idxes[r,c], self.crop_H_n)
                # print(f"move {old_idx} to {new_idx}")
                # print(f"store tensor shape {shuffled_sample.size()}")
                # print(f"input tensor shape {sample.size()}")
                # print(f"store region ({new_idx[0] * crop_H_size}:{(new_idx[0] + 1) * crop_H_size}, "
                #       f"{new_idx[1] * crop_W_size}:{(new_idx[1] + 1) * crop_W_size})")
                # print(f"input region ({old_idx[0] * crop_H_size}:{(old_idx[0] + 1) * crop_H_size}, "
                #       f"{old_idx[1] * crop_W_size}:{(old_idx[1] + 1) * crop_W_size})")
                shuffled_sample[:,:,new_idx[0] * crop_H_size:(new_idx[0] + 1) * crop_H_size,
                new_idx[1] * crop_W_size: (new_idx[1] + 1) * crop_W_size]= sample[:,:,
                                                                           crop_H_left_margin + old_idx[0] * crop_H_size:(old_idx[0] + 1) * crop_H_size,
                                                                           crop_W_left_margin + old_idx[1] * crop_W_size:(old_idx[1] + 1) * crop_W_size]



        return shuffled_sample
        # sample[shuffled_idx ]
        # ri = shuffled_idx // self.crop_W_n
        # ci = shuffled_idx % self.crop_W_n




class CameraShake(object):
    '''
    Subdivide tensor in spatial dimensions and shuffle the subdivided blocks.
    '''
    def __init__(self, cropShape, velocityScale, springStiffness):
        assert isinstance(cropShape, (float, tuple))
        self.crop_shape = cropShape
        self.crop_H_p = cropShape[0]
        self.crop_W_p = cropShape[1]
        self.velocityScale = velocityScale
        self.springStiff = springStiffness


    def bilinearInterp(self, img, crop_h, crop_w, xStart,xEnd, yStart, yEnd):

        y = torch.linspace(yStart, yEnd, crop_h)
        x = torch.linspace(xStart, xEnd, crop_w)

        meshy, meshx = torch.meshgrid((y, x))
        grid = torch.stack((meshx, meshy), 2)
        grid = torch.unsqueeze(grid, 0)
        img_expanded = torch.unsqueeze(img, 0)
        # output = img[:, int(yStart):int(yEnd),int(xStart):int(xEnd)]
        output = torch.nn.functional.grid_sample(img_expanded, grid, mode='bilinear')
        # output.squeeze(0)
        return output.squeeze(0)

    def createIdxGrid(self, crop_h, crop_w, xStart, xEnd, yStart, yEnd):
        y = torch.linspace(yStart, yEnd, crop_h)
        x = torch.linspace(xStart, xEnd, crop_w)

        meshy, meshx = torch.meshgrid((y, x))
        grid = torch.stack((meshx, meshy), 2)
        return grid
        # grid = torch.unsqueeze(grid, 0)
        # img_expanded = torch.unsqueeze(img, 0)
        # output = torch.nn.functional.grid_sample(img_expanded, grid, mode='bilinear')
        # return output.squeeze(0)


    @staticmethod
    def mapTo01(val, range):
        return val/range * 2 - 1;

    @staticmethod
    def sampleFromNeg1ToPos1():
        return 2* torch.rand(2) - 1

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        original_h = sample.size(2)
        original_w = sample.size(3)
        crop_H = int(self.crop_H_p * original_h)
        crop_W = int(self.crop_W_p * original_w)
        T = sample.size(0)
        # C = sample.size(1)

        offset = CameraShake.sampleFromNeg1ToPos1() * self.velocityScale #Offset from the top left corner
        velocity = CameraShake.sampleFromNeg1ToPos1() # this is the unscaled velocity

        topLeftMargin = torch.tensor([(original_h - crop_H)/2.0, (original_w - crop_W)/2.0])
        bottomRightMargin = torch.tensor([original_h, original_w]) - topLeftMargin

        grids = []
        # new_frames = []
        for i in range(T):
            # frame = sample[i, :, :, :]

            offset = offset + velocity * self.velocityScale
            offset = torch.clamp(offset, -topLeftMargin, bottomRightMargin)

            velocity = velocity + CameraShake.sampleFromNeg1ToPos1() - self.springStiff * offset
            # .int()
            startx = topLeftMargin[1] + offset[1]
            starty = topLeftMargin[0] + offset[0]

            yStart01 = CameraShake.mapTo01(starty, original_h)
            yEnd01 = CameraShake.mapTo01(starty + crop_H, original_h)
            xStart01 = CameraShake.mapTo01(startx, original_w)
            xEnd01 = CameraShake.mapTo01(startx + crop_W, original_w)
            # newFrame = self.bilinearInterp(frame, crop_H, crop_W, xStart01, xEnd01, yStart01, yEnd01)
            # new_frames.append(newFrame)
            new_grid = self.createIdxGrid(crop_H, crop_W, xStart01, xEnd01, yStart01, yEnd01)
            grids.append(new_grid)

        torch_grid = torch.stack(grids)
        output = torch.nn.functional.grid_sample(sample, torch_grid, mode='bilinear')
        #
        # new_sample = torch.stack(new_frames, dim=0)

        return output




