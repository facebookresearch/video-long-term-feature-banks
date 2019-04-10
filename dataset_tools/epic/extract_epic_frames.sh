#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

IN_DATA_DIR="../../data/epic/videos"
OUT_DATA_DIR="../../data/epic/frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*/*MP4)
do
  video_name=${video##*/}
  video_name=${video_name::-4}
  person=${video_name::-3}

  out_person_dir=${OUT_DATA_DIR}/${person}/
  mkdir -p "${out_person_dir}"

  out_name="${out_person_dir}/${video_name}_%06d.jpg"

  ffmpeg -i "${video}" -vf scale=-1:340 -r 30 -q:v 1 "${out_name}"
done
