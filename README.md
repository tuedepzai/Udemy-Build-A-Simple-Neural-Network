# Mô hình dự đoán giá xe của bạn hoạt động như thế nào?

Ngày lập: 17/10/2026. Dựa trên code và kết quả đang lưu trong `data-preparation.ipynb`.

Báo cáo này là tài liệu học cùng notebook: ưu tiên hiểu cơ chế, đọc được kết quả và tự làm vài thí nghiệm nhỏ. Các chỉ số dưới đây lấy từ đầu ra đã lưu, không phải kết quả của một lần huấn luyện mới.

## 1. Bạn đã xây dựng được gì?

Bạn đã xây dựng một mô hình nhận 8 thông tin về xe và trả về một số: giá hiện tại. Bài toán dự đoán một đại lượng bằng số như vậy gọi là **hồi quy**.

Trong cấu hình hiện tại, `Normalization → Dense(1)` không có hàm kích hoạt phi tuyến, nên đây là **hồi quy tuyến tính nhiều đầu vào**. Hiểu đơn giản: mô hình nhân mỗi thông tin với một hệ số, cộng chúng lại và thêm một số bù. Việc dùng TensorFlow không tự động làm mô hình trở thành mạng học sâu. [Cách hoạt động của Dense](https://keras.io/api/layers/core_layers/dense/)

Mô hình của bạn có thể được dùng để học cách xây một quy trình machine learning hoàn chỉnh. Kết quả trên bộ dữ liệu này chưa đủ để kết luận nó định giá tốt xe ngoài thực tế.

## 2. Một chiếc xe đi qua chương trình như thế nào?

```text
Thông tin xe X
     ↓
Chuẩn hóa từng cột bằng thống kê của tập train
     ↓
Dense(1): nhân hệ số, cộng và thêm số bù
     ↓
Giá dự đoán trên thang chuẩn hóa
     ↓
Nhân độ lệch chuẩn giá + cộng giá trung bình của train
     ↓
Giá dự đoán theo đơn vị ban đầu
```

Khi huấn luyện, chương trình còn so sánh dự đoán với giá thật đã chuẩn hóa, tính loss và điều chỉnh các hệ số. Khi `predict()`, chương trình chỉ tính đầu ra bằng các hệ số đã học.

## 3. Dữ liệu nào thực sự được đưa vào model?

Notebook lưu 1.000 dòng dữ liệu. Mỗi dòng là một mẫu xe. Với dòng:

```python
X = shuffled_data[:, 3:-1]
y = shuffled_data[:, -1]
```

bạn chọn đúng 8 cột sau cho X:

| Cột | Cách hiểu để đọc bài |
|---|---|
| `years` | Số năm sử dụng theo dữ liệu |
| `km` | Quãng đường đã chạy |
| `rating` | Điểm đánh giá |
| `condition` | Điểm tình trạng xe |
| `economy` | Chỉ số tiết kiệm nhiên liệu |
| `top speed` | Tốc độ tối đa |
| `hp` | Công suất |
| `torque` | Mô-men xoắn |

y là cột `current price`. Đơn vị tiền và định nghĩa chi tiết các thang điểm cần đối chiếu mô tả dataset; báo cáo dùng “đơn vị giá”, không tự coi là VND.

Ba cột `v.id`, `on road old`, `on road now` không nằm trong X hiện tại. Đây là lựa chọn đặc trưng của bạn, không phải thao tác tự động của TensorFlow. **Đặc trưng** là thông tin đầu vào mà bạn cho phép mô hình sử dụng.

`tf.random.shuffle(data)` đổi thứ tự cả dòng nên thông tin xe và giá tương ứng vẫn đi cùng nhau. Sau khi chia:

| Tập | Số xe | Dùng để làm gì? |
|---|---:|---|
| Train | 800 | Học thống kê chuẩn hóa và các hệ số dự đoán |
| Validation | 100 | Theo dõi kết quả, hỗ trợ chọn cách huấn luyện |
| Test | 100 | Đánh giá cuối sau khi chốt các lựa chọn |

Nếu thử nhiều cách và liên tục chọn cách tốt nhất theo test, test cũng dần tham gia vào quyết định của bạn. Khi đó cần một tập kiểm tra cuối còn chưa dùng để lựa chọn. [Nguyên tắc tách dữ liệu và tránh rò rỉ](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

## 4. Chuẩn hóa X và y làm gì?

### Một phép tính nhỏ để hiểu

Giả sử trong **tập train**, quãng đường trung bình là 100.000 km, độ lệch chuẩn là 20.000 km. Độ lệch chuẩn diễn tả mức phân tán quanh trung bình.

Xe đã chạy 120.000 km được biến đổi thành:

```text
(120.000 − 100.000) / 20.000 = 1
```

Số 1 nghĩa là cao hơn trung bình một độ lệch chuẩn. Xe 100.000 km thành 0, xe 80.000 km thành −1. Đây là ví dụ minh họa, không phải thống kê thật lấy từ dữ liệu của bạn.

Công thức duy nhất cần nhớ ở bước này:

\[
z = \frac{x-\mu}{\sigma}
\]

`μ` là trung bình, `σ` là độ lệch chuẩn. Chuẩn hóa này không ép tất cả số vào khoảng 0 đến 1; số âm và số lớn hơn 1 đều bình thường. Với `axis=-1` và X có dạng `(800, 8)`, mỗi cột có thống kê riêng. [Normalization của Keras](https://keras.io/api/layers/preprocessing_layers/numerical/normalization/)

### Hai thao tác khác nhau trong code

```python
normalizer_y.adapt(y_TRAIN)
```

Dòng này học và lưu thống kê của giá train. Nó không thay thế nội dung `y_TRAIN`.

```python
y_train_norm = normalizer_y(y_TRAIN)
```

Dòng này mới tạo các giá trị đã biến đổi. Vì vậy bạn đã sửa đúng khi truyền `normalizer_y(y_TRAIN)` vào `fit()`.

X được chuẩn hóa bên trong model vì bạn đặt `normalizer` trong `Sequential`. y được chuẩn hóa bên ngoài vì nó là đáp án để so sánh với dự đoán.

### Có làm rò rỉ dữ liệu không?

Trong notebook hiện tại, cả hai `adapt()` đều chỉ dùng train. Validation chỉ được biến đổi bằng thống kê đã có. Cách chuẩn hóa đó không dùng thống kê validation/test để huấn luyện.

Ví dụ minh họa: train có giá trung bình 300.000 và độ lệch chuẩn 100.000. Một nhãn validation 400.000 trở thành 1. Giá 400.000 không được dùng để tính lại hai số 300.000 và 100.000.

## 5. Bên trong Dense(1): mô hình thực sự học gì?

Gọi 8 đầu vào sau chuẩn hóa là `z₁` đến `z₈`. Đầu ra của lớp Dense là:

\[
\hat y_{norm}=w_1z_1+w_2z_2+\cdots+w_8z_8+b
\]

- `ŷ` đọc là “y dự đoán”.
- `w` là hệ số nhân, thường gọi là trọng số.
- `b` là số bù, còn gọi là bias.

Model của bạn học **9 số: 8 trọng số và 1 bias**. Thống kê của normalizer được tính ở bước `adapt()`, không phải 9 số được optimizer điều chỉnh khi `fit()`.

Ví dụ thu nhỏ với chỉ hai đặc trưng, giả sử các hệ số là:

```text
Giá chuẩn hóa = −0,4 × tuổi xe chuẩn hóa
                −0,2 × số km chuẩn hóa + 0,1
```

Xe có tuổi chuẩn hóa bằng 1, số km chuẩn hóa bằng 0 sẽ được dự đoán là −0,3. Nếu trung bình giá là 300.000 và độ lệch chuẩn là 100.000 thì giá dự đoán là 270.000. Đây là ví dụ tự đặt để hiểu phép tính, không phải trọng số mô hình của bạn.

Bạn có thể xem các hệ số thật bằng:

```python
weights, bias = model.layers[-1].get_weights()
names = ['years', 'km', 'rating', 'condition',
         'economy', 'top speed', 'hp', 'torque']

for name, weight in zip(names, weights[:, 0]):
    print(name, float(weight))
print('bias:', float(bias[0]))
```

Với mô hình này, tăng một đặc trưng chuẩn hóa thêm 1 và giữ các đầu vào khác cố định sẽ làm đầu ra chuẩn hóa thay đổi bằng hệ số của nó. Tuy nhiên, hệ số học từ dữ liệu không tự chứng minh quan hệ nhân quả: nó không chứng minh việc thay riêng một bộ phận xe sẽ làm giá thị trường thay đổi đúng như vậy.

## 6. Model “học” trong fit() như thế nào?

Mỗi bước huấn luyện gồm một vòng nhỏ:

```text
Lấy một nhóm xe → dự đoán → so với giá thật đã chuẩn hóa
→ tính loss → tính hướng điều chỉnh → cập nhật 9 hệ số
```

**Batch** là nhóm mẫu xử lý trong một bước. Log của bạn ghi 25 bước mỗi epoch với 800 mẫu train, phù hợp với batch mặc định 32 mẫu. **Epoch** là một lượt đi qua toàn bộ tập train. Vì vậy 100 epoch tương ứng khoảng 2.500 lần cập nhật trong lần chạy đang lưu.

**Loss** là con số được dùng làm mục tiêu khi học. Bạn chọn Huber. Với sai số `e = dự đoán − giá thật` trên thang chuẩn hóa và ngưỡng mặc định bằng 1:

\[
L(e)=
\begin{cases}
\frac12 e^2 & |e|\leq1\\
|e|-\frac12 & |e|>1
\end{cases}
\]

Không cần biến đổi dài: sai số nhỏ dùng bình phương; sai số lớn dùng biểu thức tăng tuyến tính, để các mẫu sai rất lớn có ảnh hưởng nhẹ hơn so với loss bình phương thuần túy. [Huber trong Keras](https://keras.io/api/losses/regression_losses/#huber-class)

**Gradient**, hay độ dốc theo tham số, cho biết loss thay đổi thế nào nếu nhích một hệ số. TensorFlow tự tính phần này. **Adam** dùng các độ dốc hiện tại và thông tin tích lũy để điều chỉnh hệ số. `learning_rate=0.001` điều khiển quy mô cập nhật; nó không có nghĩa mỗi bước tăng dự đoán đúng 0,001. [Adam trong Keras](https://keras.io/api/optimizers/adam/)

Bạn đang học điều khiển nên có thể liên tưởng tới việc đo sai số rồi chỉnh tham số để lần sau tốt hơn. Nhưng đây là quá trình tối ưu tham số trên dữ liệu. Nó chưa phải một bộ điều khiển xe chạy theo thời gian thực.

## 7. Vì sao trước đây predict chỉ ra khoảng 0–5?

Ở bản trước, X đã chuẩn hóa nhưng y vẫn ở hàng trăm nghìn. Lớp Dense bắt đầu với đầu ra nhỏ, còn muốn khớp giá gốc thì cần học các hệ số có độ lớn phù hợp. Với Adam 0,001, Huber mặc định và số bước đang dùng, log cũ cho thấy tiến triển quá ít.

Với Huber, khi sai số vượt ngưỡng 1, độ dốc theo dự đoán có độ lớn bằng 1. Sai 300.000 không tạo độ dốc lớn gấp 300.000 lần sai 1. Adam còn điều chỉnh độ dốc theo lịch sử của nó, nên không thể suy ra tốc độ học chỉ từ độ lớn loss.

Nhận định ở đây dựa trên code và log bạn đã lưu: sau khi đổi y sang thang chuẩn hóa, kết quả học cải thiện rõ. Điều này không có nghĩa mọi bài hồi quy đều bắt buộc chuẩn hóa y. Việc đổi thang y cũng đổi ý nghĩa thực tế của ngưỡng Huber bằng 1: bây giờ nó ứng với một độ lệch chuẩn giá train.

## 8. Predict và phép đổi về giá gốc

Đoạn hiện tại của bạn:

```python
y_pred_norm = model.predict(X_TEST)

y_pred = (
    y_pred_norm * tf.sqrt(normalizer_y.variance)
    + normalizer_y.mean
).numpy()[:, 0]
```

Đây là phép đảo của chuẩn hóa:

\[
\hat y=\hat y_{norm}\sigma_y+\mu_y
\]

`variance` là phương sai, lấy căn bậc hai để có độ lệch chuẩn. Bạn giữ kết quả dạng `(100, 1)` trong lúc tính, rồi mới lấy cột duy nhất để thu được `(100,)`. Cách này tránh lỗi trước đây khiến kết quả cuối chỉ còn một dự đoán.

Khi dự đoán xe mới, đưa đủ 8 giá trị **gốc**, đúng thứ tự cột vào model. Đừng chuẩn hóa X thêm lần nữa trước `predict()`, vì model đã chứa normalizer của X. Sau đó đổi đầu ra về giá gốc bằng thống kê y đã lưu.

## 9. Kết quả hiện tại nói lên điều gì?

| Chỉ số | Giá trị trong kết quả đang lưu | Đơn vị |
|---|---:|---|
| Validation RMSE, epoch đầu | Khoảng 1,4437 | Thang y chuẩn hóa |
| Validation RMSE, epoch cuối | Khoảng 0,3433 | Thang y chuẩn hóa |
| Train RMSE, epoch cuối | Khoảng 0,3337 | Thang y chuẩn hóa |
| Test RMSE, tính lại từ 100 cặp đã in | Khoảng 43.349 | Đơn vị giá gốc |
| Test MAE, tính lại từ 100 cặp đã in | Khoảng 35.749 | Đơn vị giá gốc |

Hai chỉ số test được tính từ những số đã in trong notebook nên là giá trị xấp xỉ. Mình không chạy lại training. Để tính trực tiếp trong phiên notebook của bạn:

```python
y_test_np = y_TEST[:, 0].numpy()
errors = y_pred - y_test_np
print('Test MAE:', np.mean(np.abs(errors)))
print('Test RMSE:', np.sqrt(np.mean(errors ** 2)))
```

**MAE** là trung bình độ lớn sai số. Ở lần chạy đang lưu, nó nói rằng sai lệch tuyệt đối trung bình trên 100 xe test xấp xỉ 35.749 đơn vị giá.

**RMSE** lấy căn của trung bình bình phương sai số:

\[
RMSE=\sqrt{\frac{e_1^2+\cdots+e_n^2}{n}}
\]

Nó nhạy hơn với sai số lớn. RMSE 43.349 không có nghĩa mọi xe đều sai đúng 43.349, và RMSE chuẩn hóa 0,3433 không có nghĩa “đúng 65,67%”.

Xe đầu trong test có giá thật 271.010,5 và dự đoán 278.448,4, chênh khoảng 7.438. Đó chỉ là một ví dụ; dùng toàn bộ tập test mới đánh giá được kết quả chung.

Train và validation RMSE cuối khá gần nhau. Đây là dấu hiệu thuận lợi trong lần chia dữ liệu này, nhưng chưa đủ để kết luận mô hình tốt trên mọi dữ liệu mới. Chưa có kết quả so sánh với một cách dự đoán đơn giản hay nhiều lần chia tập.

### Các hình đã xuất từ notebook

![Validation loss](validation-loss.png)

*Loss trên validation. Đường giảm cho thấy mục tiêu này cải thiện; mức giảm và đơn vị vẫn cần được đọc cùng các chỉ số.*

![Validation RMSE](validation-rmse.png)

*RMSE trên validation theo thang y chuẩn hóa. Khi đường gần nằm ngang, chạy thêm nhiều epoch chưa chắc đem lại cải thiện đáng kể.*

![Giá dự đoán và giá thật](test-predictions.png)

*100 mẫu test theo thứ tự sau xáo trộn. Trục ngang là chỉ số mẫu, không phải thời gian. Hai cột gần nhau về chiều cao nghĩa là dự đoán gần giá thật. Với 100 mẫu và cột mảnh, hình khá dày; xem 20 mẫu một lần sẽ dễ đối chiếu hơn.*

## 10. Ba thí nghiệm nhỏ để bạn tự hiểu sâu hơn

Làm từng việc, ghi lại điều mình dự đoán trước khi chạy. Mỗi lần chỉ thay một yếu tố để biết nguyên nhân khác biệt.

### A. Tự tính một lần chuẩn hóa

Cho trung bình giá 300.000, độ lệch chuẩn 100.000. Bạn tự tính giá 250.000 thành bao nhiêu trên thang chuẩn hóa, rồi đảo lại. Không dùng model. Mục tiêu là giải thích được số âm có nghĩa gì, không cần làm nhiều bài tính.

### B. So với cách “xe nào cũng đoán giá trung bình train”

Trước hết dùng validation để so sánh các lựa chọn:

```python
y_val_np = y_VAL[:, 0].numpy()
baseline_price = float(tf.reduce_mean(y_TRAIN).numpy())
baseline_rmse = np.sqrt(np.mean((y_val_np - baseline_price) ** 2))

model_rmse = 0.3432987928390503 * float(
    tf.sqrt(normalizer_y.variance).numpy().item()
)
print('Đoán giá trung bình — validation RMSE:', baseline_rmse)
print('Model hiện tại — validation RMSE:', model_rmse)
```

Số `0.3432987928390503` chỉ dành cho lần chạy được báo cáo. Nếu train lại, lấy RMSE mới từ `model.evaluate(...)` rồi đổi về giá gốc. Vì chỉ đổi thang tuyến tính, RMSE giá gốc bằng RMSE chuẩn hóa nhân độ lệch chuẩn y train.

**Câu hỏi:** mô hình có tốt hơn cách đoán trung bình không? Nếu chưa, bạn chưa có bằng chứng rằng dùng 8 thông tin đang giúp ích. “Baseline” là tên gọi cho mốc so sánh đơn giản này.

### C. Thay một thông tin, quan sát dự đoán

Giữ nguyên các thông tin của một xe validation, chỉ tăng `km` thêm 10.000 rồi predict lại. Với mô hình tuyến tính này, cùng một mức tăng km sẽ tạo cùng một mức thay đổi giá dự đoán khi các đầu vào khác giữ nguyên. Bạn thử kiểm tra điều đó với hai xe khác nhau.

Không cần thêm mạng lớn ngay. Nếu muốn thí nghiệm có thể lặp lại, đặt `tf.keras.utils.set_random_seed(42)` trước khi xáo trộn và tạo model; môi trường chạy vẫn có thể ảnh hưởng một số kết quả số học.

## 11. Liên hệ với hướng robotics của bạn

Sau bài này, một bước nối phù hợp là học mô hình thực nghiệm giữa lệnh motor và tốc độ bánh xe. Bạn đo vài cặp lệnh–tốc độ rồi thử khớp quan hệ `v ≈ a·u + b` trong một vùng hoạt động hẹp. `u` là lệnh motor, `v` là tốc độ đo được.

Đây là cách dùng cùng ý tưởng học hệ số từ dữ liệu cho một vấn đề cơ điện tử. Motor thật có ma sát, vùng lệnh chưa quay, tải và pin thay đổi nên mô hình đơn giản có giới hạn. Bước đầu chỉ cần giải thích được dữ liệu vào, đáp án và sai số; chưa cần tự động điều khiển bằng mô hình đó.

## 12. Tài liệu nên đọc theo thứ tự

Chọn một tài liệu để thực hành trước, không cần mở hết cùng lúc.

| Thứ tự | Tài liệu | Đọc phần nào, để làm gì? |
|---|---|---|
| 1 | [TensorFlow: Basic regression — Predict fuel efficiency](https://www.tensorflow.org/tutorials/keras/regression) | Phần Normalization và Linear regression with multiple inputs. Bài dự đoán mức tiêu thụ nhiên liệu của xe, gần với notebook của bạn. Đối chiếu từng khối với code mình đã viết. |
| 2 | [Google Machine Learning Crash Course: Linear regression](https://developers.google.com/machine-learning/crash-course/linear-regression) | Đọc về trọng số, bias và loss. Dùng đồ thị, ví dụ để hiểu trước khi quan tâm đến phép đạo hàm. |
| 3 | [Keras: Normalization](https://keras.io/api/layers/preprocessing_layers/numerical/normalization/) | Tra ý nghĩa `axis`, `adapt()` và công thức biến đổi; kiểm tra lại bằng một ví dụ số nhỏ của bạn. |
| 4 | [scikit-learn: Common pitfalls — Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) | Đọc để phân biệt học thống kê từ train và áp dụng phép biến đổi lên validation/test. Chưa cần học toàn bộ thư viện. |

Khi cần tra riêng một dòng code: [Dense](https://keras.io/api/layers/core_layers/dense/), [Huber](https://keras.io/api/losses/regression_losses/#huber-class), [Adam](https://keras.io/api/optimizers/adam/).

Một buổi học ngắn có thể gồm: 10 phút đọc mục 4–5 của báo cáo, 10 phút tự tính một ví dụ chuẩn hóa, 15 phút in trọng số và thử thay km, rồi 5 phút ghi lại điều mình hiểu. Dấu hiệu tiến bộ cần tìm là bạn tự giải thích được **đầu vào → phép tính → sai số → cập nhật**, thay vì chỉ chạy được notebook.
