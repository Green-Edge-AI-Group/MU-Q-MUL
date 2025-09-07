import utils
import unlearn
import torch


if __name__=='__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(int(0))
        device = torch.device(f"cuda:{int(0)}")
    else:
        device = torch.device("cpu")
    checkpoint = utils.load_checkpoint(device, './save/class0/Qresnet18/cifar100/4w4a', 'retrain')
    evaluation_result = checkpoint.get("evaluation_result")
    print(evaluation_result)