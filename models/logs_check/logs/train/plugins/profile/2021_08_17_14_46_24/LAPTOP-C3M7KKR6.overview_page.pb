?	?wG?=W?@?wG?=W?@!?wG?=W?@	????????????!??????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?wG?=W?@zVҊo?*@A'???8Q?@Y?????%@rEagerKernelExecute 0*	4333kOA2
HIterator::Model::MaxIntraOpParallelism::Prefetch::ForeverRepeat::BatchV2X?2ıpo@!b??S??X@)?٬?\o@1_??tJ>X@:Preprocessing2?
QIterator::Model::MaxIntraOpParallelism::Prefetch::ForeverRepeat::BatchV2::Shuffle?|a2U@!??q??i??)?|a2U@1??q??i??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?镲q@!'7?????)?C?l??@1??%9*??:Preprocessing2F
Iterator::ModelB?f???@!??r???)Dio?????1˙????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch1?*????!
K??p???)1?*????1
K??p???:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ForeverRepeat?? ?r?o@!?_?7ʑX@)?C??????1?W??ǯ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??????I-\?O?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	zVҊo?*@zVҊo?*@!zVҊo?*@      ??!       "      ??!       *      ??!       2	'???8Q?@'???8Q?@!'???8Q?@:      ??!       B      ??!       J	?????%@?????%@!?????%@R      ??!       Z	?????%@?????%@!?????%@b      ??!       JCPU_ONLYY??????b q-\?O?X@Y      Y@q?>vz?v??"?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 