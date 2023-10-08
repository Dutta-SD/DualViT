from torch.utils.tensorboard import SummaryWriter

from vish.model.tp.dual import TPDualVit

net =

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter("runs/tpdualvit")

writer.add_graph(net, images)
writer.close()