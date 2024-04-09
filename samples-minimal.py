# %%
import numpy as np
import torch
import torch.nn.functional as F
from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG, RotateNTurns

# Setup Model
model = get_model('hg_furukawa_original', 51)

n_classes = 44
split = [21, 12, 11]
model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
checkpoint = torch.load('model_best_val_loss_var.pkl')

model.load_state_dict(checkpoint['model_state'])
model.eval()
model.cuda()
print("Model loaded.")

# Test the model
data_folder = 'data/cubicasa5k/'
data_file = 'test.txt'
normal_set = FloorplanSVG(data_folder, data_file, format='txt', original_size=True)
data_loader = torch.utils.data.DataLoader(normal_set, batch_size=1, num_workers=0)
data_iter = iter(data_loader)

val = next(data_iter)
image = val['image'].cuda()

with torch.no_grad():
    height = image.shape[2]
    width = image.shape[3]
    img_size = (height, width)
    
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    pred_count = len(rotations)
    prediction = torch.zeros([pred_count, n_classes, height, width])
    for i, r in enumerate(rotations):
        forward, back = r
        # We rotate first the image
        rot_image = RotateNTurns()(image, 'tensor', forward)
        pred = model(rot_image)
        # We rotate prediction back
        pred = RotateNTurns()(pred, 'tensor', back)
        # We fix heatmaps
        pred = RotateNTurns()(pred, 'points', back)
        # We make sure the size is correct
        pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
        # We add the prediction to output
        prediction[i] = pred[0]

prediction = torch.mean(prediction, 0, True)
rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
rooms_pred = np.argmax(rooms_pred, axis=0)

icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
icons_pred = np.argmax(icons_pred, axis=0)

print("Rooms prediction:", rooms_pred)
print("Icons prediction:", icons_pred)
