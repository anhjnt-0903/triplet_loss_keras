# triplet_loss_keras
### Train minist data with triplet loss


Hướng dẫn sử dụng triplet loss keras: https://github.com/ma1112/keras-triplet-loss

Một điểm cần chú ý:
    
    * Trong loss triplet loss tại repo: [here](https://github.com/omoindrot/tensorflow-triplet-loss), y_true và y_predict có shape khác nhau, y_true (batch_size, ), trong khi y_predict (batch_size, 1). Điều này trong Keras sẽ không đúng, bởi vì Keras không xử lý được khi hai giá trị y_true và y_predict có shape khác nhau.
    * Issue cho vấn đề này: https://github.com/omoindrot/tensorflow-triplet-loss/issues/18
    * Giải pháp: Flatten y_true thành tensor có kích thước (batch_size, 1)

Author: Tuan Anh Nguyen

Creat_at: 6/1/2020
