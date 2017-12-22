# CÁCH CHẠY

## vnTokenizer

Đây là thư viện phục vụ cho việc tokenize Tiếng Việt

Vì ta sử dụng thư viện vnTokenizer được viết bằng java nên cần chạy vnTokenizerEntryPoint để chạy java trên python

1. mở terminal tại folder VnTokenizerEntryPoint

2. `java -jar .\VnTokenizerEntryPoint.jar`

3. giữa terminal này mở

## Model sinh câu trả lời

### Tham số

1. **challenge**: đường dẫn tới tập train format babi
`challenge = 'data/babi/vi/_{}.txt'`

### Chạy
1. mở terminal tại thư mục mine
2. `python .\babi_rnn_vi.py`

### Kết quả

cho ra 2 file ở thư mục outputs:
1. model.h5
2. model_context.npy

***chi tiết xem phần tham số của Chạy ứng dụng demo***


## Chạy ứng dụng demo

### Tham số
1. ***source***: đường dẫn tới thư mục chứa 3 file input
2. các file input

|File |Mô tả|
|-----|-----|
|model.h5|mô hình sau khi train|
|model_context.npy|chứa các 3 thông tin. **word2idx**: chuyển từ index sang word. **story_maxlen**: độ dài tối đa của văn bản. **query_maxlen**: độ dài tối đa của chuỗi truy vấn|
|documents.txt|chứa các văn bản|
### Chạy
`python .\app.py`

- nhập câu có dấu **?** để hỏi
- nhập **bye** để thoát
