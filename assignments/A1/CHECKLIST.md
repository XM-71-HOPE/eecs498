# A1 进度清单

一共 **28 个实现**（20 个在 `pytorch101.py`，8 个在 `knn.py`）。
另有两道在 notebook 的 markdown 里手写方程的小题，别忘了。

用法：做完一个划掉一个。行号对应 `.py` 文件里的位置。

---

## 第一批 · 张量基础与索引（12 个）

### 张量创建（`pytorch101.py`）

- [ ] 27 · `create_sample_tensor`
- [ ] 61 · `mutate_tensor`
- [ ] 88 · `count_tensor_elements`
- [ ] 110 · `create_tensor_of_pi`
- [ ] 135 · `multiples_of_ten`

### 索引与切片

- [ ] 173 · `slice_indexing_practice`
- [ ] 215 · `slice_assignment_practice`
- [ ] 242 · `shuffle_cols`
- [ ] 271 · `reverse_rows`
- [ ] 299 · `take_one_elem_per_col`
- [ ] 326 · `make_one_hot`
- [ ] 362 · `sum_positive_entries`

---

## 第二批 · 形状与运算（8 个）

### 变形（`pytorch101.py`）

- [ ] 392 · `reshape_practice`
- [ ] 430 · `zero_row_min`

### 运算与向量化

- [ ] 485 · `batched_matrix_multiply_loop`
- [ ] 515 · `batched_matrix_multiply_noloop`
- [ ] 549 · `normalize_columns`

### 广播与收尾

- [ ] 595 · `mm_on_gpu`
- [ ] 627 · `challenge_mean_tensors`
- [ ] 666 · `challenge_get_uniques`

---

## 第三批 · k-NN（8 个，全在 `knn.py`）

- [ ] 56 · `compute_distances_two_loops`
- [ ] 100 · `compute_distances_one_loop`
- [ ] 146 · `compute_distances_no_loops`
- [ ] 193 · `predict_labels`
- [ ] 217 · `KNN.__init__`
- [ ] 240 · `KNN.predict`
- [ ] 311 · `knn_cross_validate`
- [ ] 362 · `knn_get_best_k`

---

## 两个提醒

1. `mm_on_gpu` 在纯 CPU 机器上会走不到 GPU 分支，照着 docstring 写就好，不用管跑不跑得起来。
2. 最后那个 `make_a1_submission(...)` 的 cell 直接跳过，校外没有 Autograder。
