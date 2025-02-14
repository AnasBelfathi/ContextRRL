from prettytable import PrettyTable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


import torch
import random
import json
import time
import models
import numpy as np
import os

from eval import eval_model
from utils import tensor_dict_to_gpu, tensor_dict_to_cpu, ResultWriter, get_num_model_parameters, print_model_parameters
from task import Task, Fold
import gc
import copy

class SentenceClassificationTrainer:
    '''Trainer for baseline model and also for Sequantial Transfer Learning. '''
    def __init__(self, device, config, task: Task, result_writer:ResultWriter):

        # device: The computational device (CPU or GPU).
        # config: The configuration dictionary containing hyperparameters (e.g., learning rate, max epochs).
        # result_writer: A utility for logging results.
        # cur_result: A dictionary to store the current results (e.g., task name, configuration).
        # labels: The list of labels from the Task object (e.g., rhetorical roles).
        # task: The Task object, which manages data and metadata (e.g., dataset path, labels).

        self.device = device
        self.config = config
        self.result_writer = result_writer
        self.cur_result = dict()
        self.cur_result["task"] = task.task_name
        self.cur_result["config"] = config

        self.labels = task.labels
        self.task = task

    def write_results(self, fold_num, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion):
        # Purpose: Logs the current results for the given fold and epoch, including metrics and confusion matrices for both the validation (dev) and test sets.
        self.cur_result["fold"] = fold_num
        self.cur_result["epoch"] = epoch
        self.cur_result["train_duration"] = train_duration
        self.cur_result["dev_metrics"] = dev_metrics
        self.cur_result["dev_confusion"] = dev_confusion
        self.cur_result["test_metrics"] = test_metrics
        self.cur_result["test_confusion"] = test_confusion

        self.result_writer.write(json.dumps(self.cur_result))


    def run_training_for_fold(self, fold_num, fold: Fold, initial_model=None, return_best_model=False, path=None):

        self.result_writer.log(f'device: {self.device}')

        train_batches, dev_batches, test_batches = fold.train, fold.dev, fold.test

        self.result_writer.log(f"fold: {fold_num}")
        self.result_writer.log(f"train batches: {len(train_batches)}")
        self.result_writer.log(f"dev batches: {len(dev_batches)}")
        self.result_writer.log(f"test batches: {len(test_batches)}")

        # instantiate model per reflection
        if initial_model is None:
            model = getattr(models, self.config["model"])(self.config, [self.task])
        else:
            self.result_writer.log("Loading weights from initial model....")
            model = copy.deepcopy(initial_model)
            # for transfer learning do not transfer the output layer
            model.reinit_output_layer([self.task], self.config)

        self.result_writer.log("Model: " + model.__class__.__name__)
        self.cur_result["model"] = model.__class__.__name__

        model.to(self.device)

        max_train_epochs = self.config["max_epochs"]
        lr = self.config["lr"]
        max_grad_norm = 1.0

        self.result_writer.log(f"Number of model parameters: {get_num_model_parameters(model)}")
        self.result_writer.log(f"Number of model parameters bert: {get_num_model_parameters(model.bert)}")
        self.result_writer.log(f"Number of model parameters word_lstm: {get_num_model_parameters(model.word_lstm)}")
        self.result_writer.log(f"Number of model parameters attention_pooling: {get_num_model_parameters(model.attention_pooling)}")
        self.result_writer.log(f"Number of model parameters sentence_lstm: {get_num_model_parameters(model.sentence_lstm)}")
        self.result_writer.log(f"Number of model parameters crf: {get_num_model_parameters(model.crf)}")
        print_model_parameters(model)

        # for feature based training use Adam optimizer with lr decay after each epoch (see Jin et al. Paper)
        optimizer = Adam(model.parameters(), lr=lr)
        epoch_scheduler = StepLR(optimizer, step_size=1, gamma=self.config["lr_epoch_decay"])

        best_dev_result = 0.0
        early_stopping_counter = 0
        epoch = 0
        early_stopping = self.config["early_stopping"]
        best_model = None
        optimizer.zero_grad()
        while epoch < max_train_epochs and early_stopping_counter < early_stopping:
            epoch_start = time.time()

            self.result_writer.log(f'training model for fold {fold_num} in epoch {epoch} ...')

            random.shuffle(train_batches)
            # train model
            model.train()
            for batch_num, batch in enumerate(train_batches):
                # move tensor to gpu
                tensor_dict_to_gpu(batch, self.device)

                # Ensure labels are for the target sentence
                target_labels = batch["label_ids"] * batch["target_sentences"]  # Mask labels for non-target sentences
                # print(target_labels)
                # print(batch["target_sentences"])  # Should be a tensor with values 0 or 1
                # print(batch["doc_name"])  # Should display document names

                #Debugging: Inspect target sentences and labels
                # print(f"\n[DEBUG] Batch {batch_num}")
                # print(f"Doc name: \n{batch['doc_name']}")
                # print(f"Target Sentences Mask:\n{batch['target_sentences']}")  # Vérifie le masque des phrases cibles
                # print(f"Label IDs:\n{batch['label_ids']}")  # Vérifie les labels
                # target_labels = batch["label_ids"] * batch["target_sentences"]  # Filtrage des labels des phrases cibles
                # print(f"Filtered Target Labels:\n{target_labels}")  # Vérifie les labels après application du masque
                # print(f"sentence mask:\n{batch['sentence_mask']}")





                output = model(
                    batch=batch,
                    labels=target_labels
                )
                # print(f"output:\n{output}")
                loss = output["loss"]
                loss = loss.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # move batch to cpu again to save gpu memory
                tensor_dict_to_cpu(batch)

                if batch_num % 100 == 0:
                    self.result_writer.log(f"Loss in fold {fold_num}, epoch {epoch}, batch {batch_num}: {loss.item()}")

            train_duration = time.time() - epoch_start

            epoch_scheduler.step()

            # evaluate model
            results={}
            self.result_writer.log(f'evaluating model...')




            dev_metrics, dev_confusion,labels_dict, _ = eval_model(model, dev_batches, self.device, self.task)
            results['dev_metrics']=dev_metrics
            results['dev_confusion'] = dev_confusion
            results['labels_dict'] = labels_dict
            results['classification_report']=_


            if dev_metrics[self.task.dev_metric] > best_dev_result:
                if return_best_model:
                    best_model = copy.deepcopy(model)
                best_dev_result = dev_metrics[self.task.dev_metric]
                early_stopping_counter = 0
                self.result_writer.log(f"New best dev {self.task.dev_metric} {best_dev_result}!")
                results={}
                test_metrics, test_confusion,labels_dict,_ = eval_model(model, test_batches, self.device, self.task)
                # Save predictions with true labels and logits
                predictions_file = f"{path}/{self.task.task_name}_test_predictions.json"
                with open(predictions_file, "w") as f:
                    json.dump(labels_dict, f, indent=4)

                self.result_writer.log(f"Test predictions saved to {predictions_file}")

                results['dev_metrics']=dev_metrics
                results['dev_confusion'] = dev_confusion
                results['labels_dict'] = labels_dict
                results['classification_report']=_


                self.write_results(fold_num, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion)
                self.result_writer.log(
                    f'*** fold: {fold_num},  epoch: {epoch}, train duration: {train_duration}, dev {self.task.dev_metric}: {dev_metrics[self.task.dev_metric]}, test weighted-F1: {test_metrics["weighted-f1"]}, test macro-F1: {test_metrics["macro-f1"]}, test accuracy: {test_metrics["acc"]}')
            else:
                early_stopping_counter += 1
                self.result_writer.log(f'fold: {fold_num}, epoch: {epoch}, train duration: {train_duration}, dev {self.task.dev_metric}: {dev_metrics[self.task.dev_metric]}')

            epoch += 1
        return best_model



