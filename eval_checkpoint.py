# eval_checkpoint.py
import torch, os
from parameters import params
from model import SELDModel
from loss import SELDLossADPIT
from metrics import ComputeSELDResults
from data_generator import DataGenerator
from torch.utils.data import DataLoader
import utils

CKPT = "checkpoints/Yeow_MSIC_FAFS_ACS_audio_multiACCDOA_20260503_135443/best_model.pth"

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load checkpoint and restore params
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
saved_params = ckpt['params']
params.update(saved_params)

print(f"Loaded checkpoint from epoch {ckpt['epoch']+1}")
print(f"Best F1: {ckpt['best_f_score']*100:.3f}%")
print(f"Best LE: {ckpt['best_ang_err']:.2f}°")
print(f"Best RDE: {ckpt['best_rel_dist_err']:.3f}")

# Set up data
feat_folder = f"mel{params['nb_mels']}_gamma_iv_ms_dnorm"
params['feat_dir'] = os.path.join(params['root_dir'], feat_folder)
params['output_dir'] = 'outputs_eval_gru'
os.makedirs(params['output_dir'], exist_ok=True)

dev_test_dataset = DataGenerator(params=params, mode='dev_test')
dev_test_iterator = DataLoader(dataset=dev_test_dataset, 
                                batch_size=params['val_batch_size'],
                                num_workers=0, shuffle=False, drop_last=False)

first_batch = next(iter(dev_test_iterator))
in_feat_shape = first_batch[0].shape

# Load model
model = SELDModel(params=params, in_feat_shape=in_feat_shape).to(device)
model.load_state_dict(ckpt['seld_model'])
model.eval()

loss_fn = SELDLossADPIT(params=params).to(device)
metrics = ComputeSELDResults(params=params, 
                              ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev'))

# Run evaluation
output_dir = params['output_dir']
val_loss = 0
with torch.no_grad():
    for j, (feats, labels) in enumerate(dev_test_iterator):
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        val_loss += loss_fn(logits, labels).item()
        start = j * params['val_batch_size']
        end = start + logits.size(0)
        utils.write_logits_to_dcase_format(logits, params, output_dir, 
                                            dev_test_iterator.dataset.label_files[start:end])

metric_scores = metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'))
val_f, val_ang, val_dist, val_rde, val_onscreen, class_wise = metric_scores
utils.print_results(val_f, val_ang, val_dist, val_rde, val_onscreen, class_wise, params)