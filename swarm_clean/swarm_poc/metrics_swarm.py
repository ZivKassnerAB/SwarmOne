from torchmetrics.detection import MeanAveragePrecision

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        samples = utils.nested_tensor_from_tensor_list(samples)

        outputs = self(samples)

        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        current_step_results = self.format_for_evaluation(outputs, targets)

        formatted_preds = []
        formatted_targets = []

        for target in targets:
            image_id = target['image_id'].item()  # Assuming image_id is a tensor with a single value

            # Format predictions
            pred = current_step_results[image_id]
            formatted_pred = {
                'boxes': pred['boxes'],
                'scores': pred['scores'],
                'labels': pred['labels'],
            }

            # Format target
            formatted_target = {
                'boxes': target['boxes'],
                'labels': target['labels'],
            }

            formatted_preds.append(formatted_pred)
            formatted_targets.append(formatted_target)

        self.metric.update(formatted_preds, formatted_targets)

        # self.log_dict(loss_dict)
        self.log("loss_ce", loss_dict["loss_ce"])
        self.log("loss_giou", loss_dict["loss_giou"])
        self.log("loss_bbox_enc", loss_dict["loss_bbox_enc"])

        return losses

    def format_for_evaluation(self, outputs, targets):
        batch_orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        batch_results = self.postprocessors['bbox'](outputs, batch_orig_target_sizes)
        final_res = {target['image_id'].item(): output for target, output in zip(targets, batch_results)}
        return final_res


    def on_validation_epoch_end(self):
        val_metrics = self.metric.compute()

        self.log("map", val_metrics["map"])
        self.log("map_50", val_metrics["map_50"])
        self.log("map_75", val_metrics["map_75"])
        self.metric.reset()