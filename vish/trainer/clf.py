import torch
import torch.nn as nn
import torch.nn.functional as F

from vish.utils import accuracy


# TODO: Adapt for multiple outputs.Currently supports 2
@DeprecationWarning
class ImageClassificationBase(nn.Module):
    """
    Base class containing utilities for classification
    NOTE: Always send broad to fine labels
    """

    def __init__(self):
        super().__init__()
        self.max_accuracy = -1

    def _fine_samples_filter(self, broad_output, fine_output, fine_labels):
        # Freeze broad, backprop gradients only for correct samples
        correct_broad_samples_mask = torch.argmax(broad_output, dim=1) == fine_labels

        fine_output, fine_labels = (
            fine_output[correct_broad_samples_mask],
            fine_labels[correct_broad_samples_mask],
        )

        return fine_output, fine_labels

    def _train(self, batch, train_mode):
        broad_output, fine_output = 0, 0
        images, fine_labels, _ = batch
        *_, fine_output = self(images)  # Generate predictions

        if train_mode == all_modes.FINE_AND_BROAD:
            fine_output, fine_labels = self._fine_samples_filter(
                broad_output, fine_output, fine_labels
            )

        loss_fine = F.cross_entropy(fine_output, fine_labels)  # Calculate loss
        # loss_broad = F.cross_entropy(broad_output, broad_labels)  # Calculate loss
        return 0, loss_fine

    def training_step(self, batch, model_mode):
        return self._train(batch, model_mode)

    @torch.no_grad()
    def validation_step(self, batch, model_mode):
        broad_output, fine_output = 0, 0
        images, fine_labels, _ = batch
        *_, fine_output = self(images)  # Generate predictions

        if model_mode == all_modes.FINE_AND_BROAD:
            fine_output, fine_labels = self._fine_samples_filter(
                broad_output, fine_output, fine_labels
            )

        loss_fine = F.cross_entropy(fine_output, fine_labels)  # Calculate loss
        # loss_broad = F.cross_entropy(broad_output, broad_labels)  # Calculate loss

        acc_fine = accuracy(fine_output, fine_labels)
        # acc_broad = accuracy(broad_output, broad_labels)

        return {
            "val_loss_fine": loss_fine.detach(),
            "val_loss_broad": 0,
            "val_acc_fine": acc_fine,
            "val_acc_broad": 0,
        }

    def validation_epoch_end(self, outputs):
        # Fine Label Validation
        batch_losses = [x["val_loss_fine"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses

        batch_accs = [x["val_acc_fine"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies

        # # Broad Label Validation
        # batch_losses_broad = [x["val_loss_broad"] for x in outputs]
        # epoch_loss_broad = torch.stack(batch_losses_broad).mean()  # Combine losses

        # batch_accs_broad = [x["val_acc_broad"] for x in outputs]
        # epoch_acc_broad = torch.stack(batch_accs_broad).mean()  # Combine losses

        return {
            "val_loss_fine": epoch_loss.item(),
            "val_acc_fine": epoch_acc.item(),
            "val_loss_broad": 0,
            "val_acc_broad": 0,
        }

    def reset_stored_accuracy(self):
        self.max_accuracy = -1

    def _get_epoch_output_string(self, epoch, result_dict):
        return (
            """

            Epoch: {:<5}

            Training Broad Class Loss: {:.5f} Fine Class Loss: {:.5f}
            Validation Broad Class Loss: {:.5f} Accuracy: {:.5f} Fine Class Loss: {:.5f} Accuracy: {:.5f}
            Best Accuracy till now: {:.5f}
            """
        ).format(
            epoch,
            result_dict["train_loss_broad"],
            result_dict["train_loss_fine"],
            result_dict["val_loss_broad"],
            result_dict["val_acc_broad"],
            result_dict["val_loss_fine"],
            result_dict["val_acc_fine"],
            self.max_accuracy,
        )

    def epoch_end(self, epoch, result_dict, model_save_path, model_mode):
        print(self._get_epoch_output_string(epoch, result_dict))

        model_save_score_label = None

        if model_mode == all_modes.BROAD_ONLY:
            model_save_score_label = "val_acc_broad"
        else:
            model_save_score_label = "val_acc_fine"

        # TODO: This stays as is. Change to a separate function once needed
        # Keep this and for now

        if result_dict[model_save_score_label] > self.max_accuracy:
            print(
                f"{model_save_score_label} Accuracy increased: {self.max_accuracy:.5f} -> {result_dict[model_save_score_label]:.5f}"
            )
            self.max_accuracy = result_dict[model_save_score_label]
            save_path = model_save_path
            torch.save(self, save_path)
            print(f"Model Saved @ {save_path}")
