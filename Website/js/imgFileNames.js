const imgFileNames = [
    "COCO_train2014_000000581857.jpg",
    "COCO_train2014_000000581839.jpg",
    "COCO_train2014_000000581789.jpg",
    "COCO_train2014_000000581766.jpg",
    "COCO_train2014_000000581739.jpg",
    "COCO_train2014_000000581738.jpg",
    "COCO_train2014_000000581719.jpg",
    "COCO_train2014_000000581686.jpg",
    "COCO_train2014_000000581657.jpg",
    "COCO_train2014_000000581629.jpg",
    "COCO_train2014_000000581563.jpg",
    "COCO_train2014_000000581518.jpg",
    "COCO_train2014_000000581481.jpg",
    "COCO_train2014_000000581466.jpg",
    "COCO_train2014_000000581446.jpg",
    "COCO_train2014_000000581425.jpg",
    "COCO_train2014_000000581419.jpg",
    "COCO_train2014_000000581354.jpg",
    "COCO_train2014_000000581346.jpg",
    "COCO_train2014_000000581282.jpg",
    "COCO_train2014_000000581258.jpg",
    "COCO_train2014_000000581249.jpg",
    "COCO_train2014_000000581227.jpg",
    "COCO_train2014_000000581199.jpg",
    "COCO_train2014_000000581198.jpg",
    "COCO_train2014_000000581196.jpg",
    "COCO_train2014_000000581136.jpg",
    "COCO_train2014_000000581108.jpg",
    "COCO_train2014_000000581057.jpg",
    "COCO_train2014_000000581021.jpg",
    "COCO_train2014_000000581009.jpg",
    "COCO_train2014_000000580957.jpg",
    "COCO_train2014_000000580945.jpg",
    "COCO_train2014_000000580919.jpg",
    "COCO_train2014_000000580913.jpg",
    "COCO_train2014_000000580905.jpg",
    "COCO_train2014_000000580851.jpg",
    "COCO_train2014_000000580849.jpg",
    "COCO_train2014_000000580843.jpg",
    "COCO_train2014_000000580837.jpg",
    "COCO_train2014_000000580785.jpg",
    "COCO_train2014_000000580741.jpg",
    "COCO_train2014_000000580718.jpg",
    "COCO_train2014_000000580695.jpg",
    "COCO_train2014_000000580668.jpg",
    "COCO_train2014_000000580631.jpg",
    "COCO_train2014_000000580625.jpg",
    "COCO_train2014_000000580609.jpg",
    "COCO_train2014_000000580600.jpg",
    "COCO_train2014_000000580579.jpg",
    "COCO_train2014_000000580562.jpg",
    "COCO_train2014_000000580549.jpg",
    "COCO_train2014_000000580541.jpg",
    "COCO_train2014_000000580523.jpg",
    "COCO_train2014_000000580510.jpg",
    "COCO_train2014_000000580505.jpg",
    "COCO_train2014_000000580434.jpg",
    "COCO_train2014_000000580408.jpg",
    "COCO_train2014_000000580396.jpg",
    "COCO_train2014_000000580374.jpg",
    "COCO_train2014_000000580344.jpg",
    "COCO_train2014_000000580296.jpg",
    "COCO_train2014_000000580277.jpg",
    "COCO_train2014_000000580257.jpg",
    "COCO_train2014_000000580238.jpg",
    "COCO_train2014_000000580234.jpg",
    "COCO_train2014_000000580142.jpg",
    "COCO_train2014_000000580120.jpg",
    "COCO_train2014_000000580052.jpg",
    "COCO_train2014_000000580026.jpg",
    "COCO_train2014_000000580008.jpg",
    "COCO_train2014_000000579997.jpg",
    "COCO_train2014_000000579909.jpg",
    "COCO_train2014_000000579907.jpg",
    "COCO_train2014_000000579901.jpg",
    "COCO_train2014_000000579787.jpg",
    "COCO_train2014_000000579785.jpg",
    "COCO_train2014_000000579680.jpg",
    "COCO_train2014_000000579667.jpg",
    "COCO_train2014_000000579663.jpg",
    "COCO_train2014_000000579632.jpg",
    "COCO_train2014_000000579631.jpg",
    "COCO_train2014_000000579571.jpg",
    "COCO_train2014_000000579533.jpg",
    "COCO_train2014_000000579446.jpg",
    "COCO_train2014_000000579440.jpg",
    "COCO_train2014_000000579395.jpg",
    "COCO_train2014_000000579382.jpg",
    "COCO_train2014_000000579366.jpg",
    "COCO_train2014_000000579332.jpg",
    "COCO_train2014_000000579329.jpg",
    "COCO_train2014_000000579299.jpg",
    "COCO_train2014_000000579255.jpg",
    "COCO_train2014_000000579215.jpg",
    "COCO_train2014_000000579206.jpg",
    "COCO_train2014_000000579186.jpg",
    "COCO_train2014_000000579179.jpg",
    "COCO_train2014_000000579165.jpg",
    "COCO_train2014_000000579156.jpg",
    "COCO_train2014_000000579145.jpg",
    "COCO_train2014_000000579138.jpg",
    "COCO_train2014_000000579136.jpg",
    "COCO_train2014_000000579057.jpg",
    "COCO_train2014_000000579051.jpg",
    "COCO_train2014_000000578950.jpg",
    "COCO_train2014_000000578924.jpg",
    "COCO_train2014_000000578884.jpg",
    "COCO_train2014_000000578875.jpg",
    "COCO_train2014_000000578841.jpg",
    "COCO_train2014_000000578808.jpg",
    "COCO_train2014_000000578805.jpg",
    "COCO_train2014_000000578766.jpg",
    "COCO_train2014_000000578734.jpg",
    "COCO_train2014_000000578718.jpg",
    "COCO_train2014_000000578702.jpg",
    "COCO_train2014_000000578652.jpg",
    "COCO_train2014_000000578649.jpg",
    "COCO_train2014_000000578626.jpg",
    "COCO_train2014_000000578619.jpg",
    "COCO_train2014_000000578567.jpg",
    "COCO_train2014_000000578523.jpg",
    "COCO_train2014_000000578521.jpg",
    "COCO_train2014_000000578519.jpg",
    "COCO_train2014_000000578513.jpg",
    "COCO_train2014_000000578459.jpg",
    "COCO_train2014_000000578375.jpg",
    "COCO_train2014_000000578369.jpg",
    "COCO_train2014_000000578331.jpg",
    "COCO_train2014_000000578326.jpg",
    "COCO_train2014_000000578294.jpg",
    "COCO_train2014_000000578250.jpg",
    "COCO_train2014_000000578184.jpg",
    "COCO_train2014_000000578154.jpg",
    "COCO_train2014_000000578128.jpg",
    "COCO_train2014_000000578108.jpg",
    "COCO_train2014_000000578070.jpg",
    "COCO_train2014_000000578063.jpg",
    "COCO_train2014_000000578056.jpg",
    "COCO_train2014_000000578046.jpg",
    "COCO_train2014_000000578037.jpg",
    "COCO_train2014_000000578009.jpg",
    "COCO_train2014_000000578002.jpg",
    "COCO_train2014_000000577953.jpg",
    "COCO_train2014_000000577948.jpg",
    "COCO_train2014_000000577907.jpg",
    "COCO_train2014_000000577850.jpg",
    "COCO_train2014_000000577830.jpg",
    "COCO_train2014_000000577819.jpg",
    "COCO_train2014_000000577809.jpg",
    "COCO_train2014_000000577808.jpg",
    "COCO_train2014_000000577725.jpg",
    "COCO_train2014_000000577657.jpg",
    "COCO_train2014_000000577637.jpg",
    "COCO_train2014_000000577586.jpg",
    "COCO_train2014_000000577583.jpg",
    "COCO_train2014_000000577558.jpg",
    "COCO_train2014_000000577556.jpg",
    "COCO_train2014_000000577455.jpg",
    "COCO_train2014_000000577447.jpg",
    "COCO_train2014_000000577421.jpg",
    "COCO_train2014_000000577416.jpg",
    "COCO_train2014_000000577405.jpg",
    "COCO_train2014_000000577399.jpg",
    "COCO_train2014_000000577362.jpg",
    "COCO_train2014_000000577358.jpg",
    "COCO_train2014_000000577343.jpg",
    "COCO_train2014_000000577333.jpg",
    "COCO_train2014_000000577320.jpg",
    "COCO_train2014_000000577278.jpg",
    "COCO_train2014_000000577225.jpg",
    "COCO_train2014_000000577221.jpg",
    "COCO_train2014_000000577197.jpg",
    "COCO_train2014_000000577176.jpg",
    "COCO_train2014_000000577140.jpg",
    "COCO_train2014_000000577129.jpg",
    "COCO_train2014_000000577126.jpg",
    "COCO_train2014_000000577087.jpg",
    "COCO_train2014_000000577083.jpg",
    "COCO_train2014_000000576994.jpg",
    "COCO_train2014_000000576973.jpg",
    "COCO_train2014_000000576931.jpg",
    "COCO_train2014_000000576902.jpg",
    "COCO_train2014_000000576895.jpg",
    "COCO_train2014_000000576829.jpg",
    "COCO_train2014_000000576818.jpg",
    "COCO_train2014_000000576810.jpg",
    "COCO_train2014_000000576771.jpg",
    "COCO_train2014_000000576758.jpg",
    "COCO_train2014_000000576736.jpg",
    "COCO_train2014_000000576732.jpg",
    "COCO_train2014_000000576702.jpg",
    "COCO_train2014_000000576689.jpg",
    "COCO_train2014_000000576598.jpg",
    "COCO_train2014_000000576581.jpg",
    "COCO_train2014_000000576543.jpg",
    "COCO_train2014_000000576526.jpg",
    "COCO_train2014_000000576500.jpg",
    "COCO_train2014_000000576496.jpg",
    "COCO_train2014_000000576457.jpg",
    "COCO_train2014_000000576430.jpg",
    "COCO_train2014_000000576389.jpg",
    "COCO_train2014_000000576322.jpg",
    "COCO_train2014_000000576290.jpg",
    "COCO_train2014_000000576286.jpg",
    "COCO_train2014_000000576262.jpg",
    "COCO_train2014_000000576225.jpg",
    "COCO_train2014_000000576212.jpg",
    "COCO_train2014_000000576188.jpg",
    "COCO_train2014_000000576187.jpg",
    "COCO_train2014_000000576157.jpg",
    "COCO_train2014_000000576153.jpg",
    "COCO_train2014_000000576138.jpg",
    "COCO_train2014_000000576128.jpg",
    "COCO_train2014_000000576098.jpg",
    "COCO_train2014_000000576040.jpg",
    "COCO_train2014_000000575980.jpg",
    "COCO_train2014_000000575961.jpg",
    "COCO_train2014_000000575955.jpg",
    "COCO_train2014_000000575949.jpg",
    "COCO_train2014_000000575873.jpg",
    "COCO_train2014_000000575826.jpg",
    "COCO_train2014_000000575756.jpg",
    "COCO_train2014_000000575743.jpg",
    "COCO_train2014_000000575711.jpg",
    "COCO_train2014_000000575704.jpg",
    "COCO_train2014_000000575703.jpg",
    "COCO_train2014_000000575701.jpg",
    "COCO_train2014_000000575649.jpg",
    "COCO_train2014_000000575641.jpg",
    "COCO_train2014_000000575627.jpg",
    "COCO_train2014_000000575594.jpg",
    "COCO_train2014_000000575574.jpg",
    "COCO_train2014_000000575526.jpg",
    "COCO_train2014_000000575519.jpg",
    "COCO_train2014_000000575502.jpg",
    "COCO_train2014_000000575490.jpg",
    "COCO_train2014_000000575461.jpg",
    "COCO_train2014_000000575421.jpg",
    "COCO_train2014_000000575417.jpg",
    "COCO_train2014_000000575294.jpg",
    "COCO_train2014_000000575284.jpg",
    "COCO_train2014_000000575220.jpg",
    "COCO_train2014_000000575055.jpg",
    "COCO_train2014_000000575049.jpg",
    "COCO_train2014_000000574983.jpg",
    "COCO_train2014_000000574961.jpg",
    "COCO_train2014_000000574957.jpg",
    "COCO_train2014_000000574870.jpg",
    "COCO_train2014_000000574857.jpg",
    "COCO_train2014_000000574829.jpg"
];
