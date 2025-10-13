function handleFileChange(event) {
    const file = event.target.files[0];
    const previewImage = document.getElementById('previewImage');
    const noImageText = document.getElementById('noImageText');
    const applyModelBtn = document.getElementById('applyModelBtn');
    
    if (file) {
        // Kiểm tra định dạng file
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!allowedTypes.includes(file.type)) {
            alert('Vui lòng chọn file ảnh định dạng JPG, JPEG hoặc PNG!');
            return;
        }
        
        // Kiểm tra kích thước file (giới hạn 10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert('File ảnh quá lớn! Vui lòng chọn file nhỏ hơn 10MB.');
            return;
        }
        
        const reader = new FileReader();
        reader.onloadend = function() {
            // Hiển thị ảnh preview
            previewImage.src = reader.result;
            previewImage.style.display = 'block';
            noImageText.style.display = 'none';
            
            // Enable apply model button
            if (applyModelBtn) {
                applyModelBtn.disabled = false;
            }
        };
        reader.readAsDataURL(file);
    } else {
        // Ẩn ảnh preview
        previewImage.src = '';
        previewImage.style.display = 'none';
        noImageText.style.display = 'block';
        
        // Disable apply model button
        if (applyModelBtn) {
            applyModelBtn.disabled = true;
        }
    }
}

// Hàm apply model - gọi API thật
function applyModel() {
    const previewImage = document.getElementById('previewImage');
    const processedImage = document.getElementById('processedImage');
    const noProcessedImageText = document.getElementById('noProcessedImageText');
    const downloadBtn = document.getElementById('downloadBtn');
    const applyModelBtn = document.getElementById('applyModelBtn');
    const fileInput = document.getElementById('fileInput');
    
    if (!previewImage.src || !fileInput.files[0]) {
        alert('Vui lòng chọn ảnh trước khi áp dụng model!');
        return;
    }
    
    // Disable button và hiển thị loading
    applyModelBtn.disabled = true;
    applyModelBtn.textContent = 'Processing...';
    
    // Tạo FormData để gửi file
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    // Lấy option được chọn
    const optionSelect = document.getElementById('optionSelect');
    const selectedOption = optionSelect ? optionSelect.value : 'imageDetection';
    formData.append('option', selectedOption);
    // Gọi API backend
    fetch('/api/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'success' && data.result_image) {
            // Hiển thị ảnh đã xử lý
            processedImage.src = data.result_image;
            processedImage.style.display = 'block';
            noProcessedImageText.style.display = 'none';
            
            // Enable download button
            downloadBtn.disabled = false;
            
            // Reset apply button
            applyModelBtn.disabled = false;
            applyModelBtn.textContent = 'Apply Model';
            
            // alert(`Xử lý ảnh hoàn thành! Thời gian: ${data.processing_time}s`);
        } else {
            throw new Error(data.message || 'Unknown error occurred');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert(`Lỗi xử lý ảnh: ${error.message}`);
        
        // Reset apply button
        applyModelBtn.disabled = false;
        applyModelBtn.textContent = '🚀 Apply Model';
    });
}

// Hàm xử lý thay đổi option
function handleOptionChange(event) {
    const selectedOption = event.target.value;
    console.log('Selected option:', selectedOption);
    
    // Có thể thêm logic khác nhau cho từng option
    if (selectedOption === 'realTimeDetection') {
        // Có thể disable một số feature cho real-time
        console.log('Real-time detection mode selected');
    } else {
        // Image detection mode
        console.log('Image detection mode selected');
    }
}

// Hàm download ảnh đã xử lý
function downloadProcessedImage() {
    const processedImage = document.getElementById('processedImage');
    
    if (processedImage && processedImage.src) {
        const link = document.createElement('a');
        link.href = processedImage.src;
        link.download = 'processed_image.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } else {
        alert('Không có ảnh đã xử lý để tải xuống!');
    }
}

const ImageDetection = {
    getImageData: function() {
        const previewImage = document.getElementById('previewImage');
        if (previewImage && previewImage.src) {
            return previewImage.src;
        }
        return null;
    },
    
    getProcessedImageData: function() {
        const processedImage = document.getElementById('processedImage');
        if (processedImage && processedImage.src) {
            return processedImage.src;
        }
        return null;
    }
};

// Event listener when page loads
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileChange);
    }
    
    // Thêm event listener cho option select
    const optionSelect = document.getElementById('optionSelect');
    if (optionSelect) {
        optionSelect.addEventListener('change', handleOptionChange);
    }
    
    // Ẩn ảnh preview ban đầu
    const previewImage = document.getElementById('previewImage');
    if (previewImage) {
        previewImage.style.display = 'none';
    }
});
