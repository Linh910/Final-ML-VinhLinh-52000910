
Bài 1: 

1)	Tìm hiểu, so sánh các phương pháp Optimizer trong huấn luyện mô hình học máy

  Optimizer  là các thuật toán điều chỉnh các tham số của mô hình trong quá trình huấn luyện để giảm thiểu hàm mất(loss function). là cơ sở để xây dựng mô hình neural network với mục đích "học" được các features hoặc pattern của dữ liệu đầu vào, từ đó có thể tìm 1 cặp weights và bias phù hợp để tối ưu hóa model.

  Optimizer được sử dụng để huấn luyện các mô hình học máy. Quá trình huấn luyện mô hình học máy bao gồm việc tìm các tham số của mô hình sao cho hàm mất mát của mô hình đạt giá trị thấp nhất. 
  
Đặc trưng của optimizer

-	Tốc độ học: là một tham số điều khiển mức độ nhanh chậm của quá trình tối ưu hóa.
-	Loại hàm mục tiêu: Một số optimizer chỉ hoạt động hiệu quả với một số loại hàm mục tiêu nhất định.
-	Kích thước tập dữ liệu: Một số optimizer có thể hoạt động hiệu quả hơn với các tập dữ liệu lớn hoặc nhỏ.
-	Số lượng tham số: Một số optimizer có thể hoạt động hiệu quả hơn với các mô hình có nhiều tham số hoặc ít tham số.


Các phương pháp của Optimizer: 

-   Gradient Descent (GD)

-   Stochastic Gradient Descent (SGD)

-   AdaGrad (Adaptive Gradient)
  
-   RMSprop

-   Adam

Optimizer có tác vụ quan trọng là tính toán gradient của hàm mất mát (loss function) theo các tham số của mô hình. Gradient là vector đạo hàm riêng của hàm mất mát theo từng tham số, và cho biết hướng và mức độ thay đổi của hàm mất mát khi các tham số thay đổi. Gradient sẽ điều chỉnh các tham số của mô hình để di chuyển mô hình gần hơn đến điểm tối ưu của hàm mất mát.

Gredient Descen(GD)

Gredient Descen(GD) là giảm dần độ dốc. Nó sẽ chọn 1 nghiệm ngẫu nhiên cứ sau mỗi vòng lặp (hay epoch) thì cho nó tiến dần đến điểm cần tìm.

Mục tiêu của GD là điều chỉnh các tham số của mô hình dựa trên đạo hàm của hàm mất mát để tìm giá trị nhỏ nhất của hàm mất mát đó.
GD hoạt động bằng cách bắt đầu từ một điểm khởi đầu ngẫu nhiên và sau đó cập nhật các tham số của hàm theo hướng giảm dần của gradient hàm tại điểm hiện tại.
Đặc trưng của Gradient Descent:
-	Đạo hàm: GD sử dụng đạo hàm của hàm mất mát để xác định hướng và tốc độ cập nhật các tham số. Đạo hàm đo lường sự thay đổi của hàm mất mát khi các tham số được điều chỉnh.

-	Hướng giảm: GD điều chỉnh các tham số theo hướng giảm của đạo hàm. Nếu đạo hàm dương, GD sẽ điều chỉnh các tham số giảm; ngược lại, nếu đạo hàm âm, GD sẽ điều chỉnh các tham số tăng.

-	Tốc độ học (learning rate): Là một siêu tham số quan trọng trong GD, tốc độ học quyết định độ lớn của bước cập nhật các tham số. Nếu tốc độ học quá nhỏ, quá trình hội tụ sẽ chậm; ngược lại, nếu tốc độ học quá lớn, quá trình hội tụ có thể không ổn định hoặc không hội tụ.


Độ tin cậy của GD:

Độ tin cậy của GD phụ thuộc vào các yếu tố như tốc độ học, cách lựa chọn hàm mất mát, và khởi tạo ban đầu của các tham số. Nếu tốc độ học được chọn sao cho phù hợp và hàm mất mát được thiết kế tốt, GD có thể hội tụ tới giá trị tối ưu. Tuy nhiên, GD cũng có thể bị mắc kẹt ở điểm cực tiểu cục bộ hoặc không hội tụ nếu tốc độ học không được cân nhắc cẩn thận.

•	Kiểu hàm mất mát: Một số hàm mất mát có thể khiến GD dễ bị mắc kẹt trong các điểm cục bộ hơn các hàm mất mát khác.

•	Kích thước dữ liệu: GD có thể hoạt động tốt hơn với các tập dữ liệu lớn hơn.

•	Số lượng tham số: GD có thể hoạt động kém hiệu quả hơn với các mô hình có nhiều tham số.

•	Giá trị của learning rate: Giá trị của learning rate quá cao hoặc quá thấp có thể khiến GD không hội tụ.


Giải thuật GD:

-	Khởi tạo các tham số ban đầu.
-	Lặp lại cho đến khi đạt được điều kiện dừng (ví dụ: số lần lặp tối đa hoặc đạt đến giá trị mất mát nhỏ hơn một ngưỡng):
-	Tính toán đạo hàm của hàm mất mát theo từng tham số.
-	Cập nhật các tham số theo
  
Công thức : xnew = xold – learning_rate*gradient(x)

•	xnew : là giá trị mới của tham số,

•	xold : là giá trị hiện tị của tham số,

•	learning_rate : là hệ số học và gradient là gradient tính toán được.

-	Trả về giá trị tối ưu của các tham số.

Code:

    function gradient_descent(f, x0, learning_rate, max_iters)
        for i = 1 to max_iters
            g = gradient(f, x_i)
            x_i = x_i - learning_rate * g  
        end for
        return x_i
    end function

Trong đó:

•	f là hàm cần tối ưu hóa

•	x0 là điểm khởi đầu

•	learning_rate là tốc độ học

•	max_iters là số lượng vòng lặp tối đa

Trong mỗi vòng lặp, GD tính toán gradient của hàm tại điểm hiện tại x_i. Sau đó, GD cập nhật các tham số của hàm theo hướng giảm dần của gradient. Quá trình này được lặp lại cho đến khi đạt được điều kiện dừng, chẳng hạn như số lượng vòng lặp tối đa hoặc độ giảm của gradient nhỏ hơn một giá trị nhất định.


Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent (SGD) là một biến thể của Gradient Descent (GD) được sử dụng để tối ưu hóa các hàm khả vi. SGD cập nhật các tham số của hàm dựa trên một mẫu ngẫu nhiên của tập dữ liệu. SGD có thể giúp GD tránh bị mắc kẹt trong các điểm cục bộ.


Đặc trưng của Stochastic Gradient Descent (SGD):
-	Mini-batch: SGD sử dụng một mini-batch để tính toán đạo hàm. Kích thước của mini-batch được lựa chọn trước và thường là một giá trị nhỏ, ví dụ như 32 hoặc 64.

-	Ngẫu nhiên: SGD lựa chọn ngẫu nhiên một mini-batch từ dữ liệu huấn luyện trong mỗi bước cập nhật. Nó đảm bảo rằng mỗi mẫu dữ liệu có cơ hội được sử dụng và giúp tránh việc rơi vào điểm cực tiểu cục bộ.

-	Tốc độ học (learning rate): Tương tự như GD, SGD cũng sử dụng tốc độ học để điều chỉnh kích thước bước cập nhật các tham số. Tuy nhiên, trong SGD, tốc độ học thường được thiết lập nhỏ hơn do tính ngẫu nhiên của mini-batch có thể làm dao động quá trình hội tụ.


Độ tin cậy của Stochastic Gradient Descent (SGD):

SGD có thể hội tụ tới giá trị tối ưu nhưng không đảm bảo tìm được điểm cực tiểu toàn cục. Do sự ngẫu nhiên của việc lựa chọn mini-batch và đôi khi có thể giúp tránh việc rơi vào điểm cực tiểu cục bộ. Tuy nhiên, SGD có thể dao động quanh vùng tối ưu và không ổn định hơn so với GD truyền thống.

•	Kiểu hàm mất mát: Một số hàm mất mát có thể khiến SGD dễ bị mắc kẹt trong các điểm cục bộ hơn các hàm mất mát khác.

•	Kích thước dữ liệu: SGD có thể hoạt động tốt hơn với các tập dữ liệu lớn hơn.

•	Số lượng tham số: SGD có thể hoạt động kém hiệu quả hơn với các mô hình có nhiều tham số.

•	Giá trị của learning rate: Giá trị của learning rate quá cao hoặc quá thấp có thể khiến SGD không hội tụ.


Giải thuật (SGD):
-	Khởi tạo các tham số ban đầu.
-	Lặp lại cho đến khi đạt được điều kiện dừng (ví dụ: số lần lặp tối đa hoặc đạt đến giá trị mất mát nhỏ hơn một ngưỡng):
-	Lựa chọn ngẫu nhiên một mini-batch từ dữ liệu huấn luyện.
-	Tính toán đạo hàm của hàm mất mát dựa trên mini-batch đó.
-	Cập nhật các tham số theo 
Công thức:

thamsoNew = thamsoOld - learning_rate * gradient

trong đó,

	thamsoNew là giá trị mới của tham số
	thamsoOld là giá trị hiện tại của tham số
	learning_rate là tỷ lệ học (learning rate)

-	Trả về giá trị tối ưu của các tham số.
  
Code:

    function stochastic_gradient_descent(f, x0, learning_rate, max_iters)
        for i = 1 to max_iters
            x_i = x_i - learning_rate * gradient(f, x_i, sample)
        end for
        return x_i
    end function

Trong đó:

•	f là hàm cần tối ưu hóa

•	x0 là điểm khởi đầu

•	learning_rate là tốc độ học

•	max_iters là số lượng vòng lặp tối đa


•	sample là một mẫu ngẫu nhiên của tập dữ liệu


Trong mỗi vòng lặp, SGD chọn một mẫu ngẫu nhiên từ tập dữ liệu và tính toán gradient của hàm tại điểm hiện tại x_i dựa trên mẫu ngẫu nhiên đó. Sau đó, SGD cập nhật các tham số của hàm theo hướng giảm dần của gradient. Quá trình này được lặp lại cho đến khi đạt được điều kiện dừng, chẳng hạn như số lượng vòng lặp tối đa hoặc độ giảm của gradient nhỏ hơn một giá trị nhất định.



 2) Tìm hiểu về Continual Learning và Test Production khi xây dựng một giải pháp học máy để giải quyết một bài toán nào đó

    - Continual Learning là giúp mô hình duy trì và mở rộng kiến thức của mình theo thời gian, ngăn chặn hiện tượng quên mô hình (catastrophic forgetting) khi học từ các nguồn dữ liệu mới.
   
    - Test Production là một quy trình kiểm tra và triển khai các mô hình học máy trong môi trường sản xuất.
   
  Bài 2: 
