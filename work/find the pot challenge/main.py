import torch
import torchvision
import dataset
import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train():

    batch_size = 1
    num_epochs = 10
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    print('building loaders...')
    train_set = dataset.Loader(root=r'/home/thememorydealer/Documents/Poinsettia_hackathon/data/side-view', transform=transforms)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True, num_workers=1
    )

    print('building model...')

    # MODEL
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
    num_classes = 2  # background defaults to 0
    in_features = model.roi_heads.box_predictor.cls_score.in_features #pre-trained backbone
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) #new head

    # OPTIM, LOSS and SCHEDULER
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt,
                                                   step_size=3,
                                                   gamma=0.1)
    torch.cuda.empty_cache()

    for epoch in range(0, num_epochs):

        for (step, data) in tqdm.tqdm(enumerate(train_loader)):

            #don't ask how long this took me...
            x = [data[0].squeeze()]
            y = [data[1]]
            # print(data[1])
            output = model(x, y)
            losses = sum(loss for loss in output.values())
            opt.zero_grad()
            losses.backward()
            print(losses)
            opt.step()
            lr_scheduler.step()


if __name__ == '__main__':
    train()