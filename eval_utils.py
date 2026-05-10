# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import logging 
from logging import Logger


import os

import torch
from tqdm import tqdm

from lm_eval import evaluator

def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m,  v, "±",  se])
            else:
                print()
                values.append([k, version, m,  v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()


def tasks_evaluate(model, tasks, batch_size = 4,device="cuda"):
    result_list = []
    from datetime import datetime
    # 获取当前日期和时间
    now = datetime.now()
    for task in tasks:
        filename_safe = now.strftime("%Y%m%d_%H%M%S")  # 20240405_153045
        # sample = {task:[0,1,2,3]}
        result = evaluator.simple_evaluate(model=model,tasks=task,batch_size = batch_size,device=device)
        result_list.append(result)
        print(make_table(result))
    print("---------------final results-----------------")
    for result in result_list:
        print(make_table(result))
        
