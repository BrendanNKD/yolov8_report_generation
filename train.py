import os
from ultralytics import YOLO
from config import CFG
import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8 live')
    parser.add_argument('--epoch',default =CFG.EPOCHS,type=int)
    parser.add_argument('--optimizer',default =CFG.OPTIMIZER,type=str)
    parser.add_argument('--batch_size',default =CFG.BATCH_SIZE,type=str)
    parser.add_argument('--base',default =CFG.BASE_MODEL,type=str)
    parser.add_argument('--seed',default =CFG.SEED,type=int)
    parser.add_argument('--lr',default=CFG.LR,type=int)
    parser.add_argument('--lr_factor',default=CFG.LR_FACTOR,type=int)
    parser.add_argument('--weight_decay',default=CFG.WEIGHT_DECAY,type=int)
    parser.add_argument('--dropout',default=CFG.DROPOUT,type=int)
    parser.add_argument('--fraction',default=CFG.FRACTION,type=int)
    parser.add_argument('--patience',default=CFG.PATIENCE,type=int)
    parser.add_argument('--profile',default=CFG.PROFILE,type=bool)
    parser.add_argument('--label_smoothing',default=CFG.LABEL_SMOOTHING,type=int)
    parser.add_argument('--exp',default=CFG.EXP_NAME,type=str)
    parser.add_argument('--dataset',default=CFG.CUSTOM_DATASET_DIR,type=str)
    parser.add_argument('--output',default=CFG.OUTPUT_DIR,type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    # Create a directory to save checkpoints for each epoch
    os.makedirs('checkpoint', exist_ok=True)

    ### Load pre-trained YOLO model
    model = YOLO(args.base)

    ### train
    model.train(
        data = os.path.join(args.output, 'data.yaml'),
        task = 'detect',
        imgsz = 640,
        epochs = args.epoch,
        batch = args.batch_size,
        optimizer = args.optimizer,
        lr0 = args.lr,
        lrf = args.lr_factor,
        weight_decay = args.weight_decay,
        dropout = args.dropout,
        fraction = args.fraction,
        patience = args.patience,
        profile = args.profile,
        label_smoothing = args.label_smoothing,
        name = f'{args.base}_{args.exp}',
        seed = args.seed,  
        val = True,
        amp = True,    
        exist_ok = True,
        resume = False,
        device = 'cpu',
        verbose = False,
        save_dir='checkpoint',  # Specify the directory for saving checkpoints
    )

if __name__ == "__main__":
    main()