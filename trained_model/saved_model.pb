??.
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??+
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:d@*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:@*
dtype0
?
layer_normalization_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namelayer_normalization_31/gamma
?
0layer_normalization_31/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_31/gamma*
_output_shapes
:@*
dtype0
?
layer_normalization_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_31/beta
?
/layer_normalization_31/beta/Read/ReadVariableOpReadVariableOplayer_normalization_31/beta*
_output_shapes
:@*
dtype0
{
dense_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_38/kernel
t
#dense_38/kernel/Read/ReadVariableOpReadVariableOpdense_38/kernel*
_output_shapes
:	?*
dtype0
r
dense_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_38/bias
k
!dense_38/bias/Read/ReadVariableOpReadVariableOpdense_38/bias*
_output_shapes
:*
dtype0
?
layer_normalization_34/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namelayer_normalization_34/gamma
?
0layer_normalization_34/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_34/gamma*
_output_shapes
:*
dtype0
?
layer_normalization_34/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_34/beta
?
/layer_normalization_34/beta/Read/ReadVariableOpReadVariableOplayer_normalization_34/beta*
_output_shapes
:*
dtype0
z
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_39/kernel
s
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes

:*
dtype0
r
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_39/bias
k
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
<transformer_block_9/multi_head_self_attention_9/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*M
shared_name><transformer_block_9/multi_head_self_attention_9/query/kernel
?
Ptransformer_block_9/multi_head_self_attention_9/query/kernel/Read/ReadVariableOpReadVariableOp<transformer_block_9/multi_head_self_attention_9/query/kernel*
_output_shapes

:@*
dtype0
?
:transformer_block_9/multi_head_self_attention_9/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:transformer_block_9/multi_head_self_attention_9/query/bias
?
Ntransformer_block_9/multi_head_self_attention_9/query/bias/Read/ReadVariableOpReadVariableOp:transformer_block_9/multi_head_self_attention_9/query/bias*
_output_shapes
:*
dtype0
?
:transformer_block_9/multi_head_self_attention_9/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*K
shared_name<:transformer_block_9/multi_head_self_attention_9/key/kernel
?
Ntransformer_block_9/multi_head_self_attention_9/key/kernel/Read/ReadVariableOpReadVariableOp:transformer_block_9/multi_head_self_attention_9/key/kernel*
_output_shapes

:@*
dtype0
?
8transformer_block_9/multi_head_self_attention_9/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8transformer_block_9/multi_head_self_attention_9/key/bias
?
Ltransformer_block_9/multi_head_self_attention_9/key/bias/Read/ReadVariableOpReadVariableOp8transformer_block_9/multi_head_self_attention_9/key/bias*
_output_shapes
:*
dtype0
?
<transformer_block_9/multi_head_self_attention_9/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*M
shared_name><transformer_block_9/multi_head_self_attention_9/value/kernel
?
Ptransformer_block_9/multi_head_self_attention_9/value/kernel/Read/ReadVariableOpReadVariableOp<transformer_block_9/multi_head_self_attention_9/value/kernel*
_output_shapes

:@*
dtype0
?
:transformer_block_9/multi_head_self_attention_9/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:transformer_block_9/multi_head_self_attention_9/value/bias
?
Ntransformer_block_9/multi_head_self_attention_9/value/bias/Read/ReadVariableOpReadVariableOp:transformer_block_9/multi_head_self_attention_9/value/bias*
_output_shapes
:*
dtype0
?
:transformer_block_9/multi_head_self_attention_9/out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:transformer_block_9/multi_head_self_attention_9/out/kernel
?
Ntransformer_block_9/multi_head_self_attention_9/out/kernel/Read/ReadVariableOpReadVariableOp:transformer_block_9/multi_head_self_attention_9/out/kernel*
_output_shapes

:*
dtype0
?
8transformer_block_9/multi_head_self_attention_9/out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8transformer_block_9/multi_head_self_attention_9/out/bias
?
Ltransformer_block_9/multi_head_self_attention_9/out/bias/Read/ReadVariableOpReadVariableOp8transformer_block_9/multi_head_self_attention_9/out/bias*
_output_shapes
:*
dtype0
z
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_36/kernel
s
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel*
_output_shapes

:*
dtype0
r
dense_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_36/bias
k
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes
:*
dtype0
z
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_37/kernel
s
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes

:*
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes
:*
dtype0
?
0transformer_block_9/layer_normalization_32/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20transformer_block_9/layer_normalization_32/gamma
?
Dtransformer_block_9/layer_normalization_32/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_9/layer_normalization_32/gamma*
_output_shapes
:*
dtype0
?
/transformer_block_9/layer_normalization_32/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_block_9/layer_normalization_32/beta
?
Ctransformer_block_9/layer_normalization_32/beta/Read/ReadVariableOpReadVariableOp/transformer_block_9/layer_normalization_32/beta*
_output_shapes
:*
dtype0
?
0transformer_block_9/layer_normalization_33/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20transformer_block_9/layer_normalization_33/gamma
?
Dtransformer_block_9/layer_normalization_33/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_9/layer_normalization_33/gamma*
_output_shapes
:*
dtype0
?
/transformer_block_9/layer_normalization_33/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_block_9/layer_normalization_33/beta
?
Ctransformer_block_9/layer_normalization_33/beta/Read/ReadVariableOpReadVariableOp/transformer_block_9/layer_normalization_33/beta*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*'
shared_nameAdam/dense_35/kernel/m
?
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes

:d@*
dtype0
?
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:@*
dtype0
?
#Adam/layer_normalization_31/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/layer_normalization_31/gamma/m
?
7Adam/layer_normalization_31/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_31/gamma/m*
_output_shapes
:@*
dtype0
?
"Adam/layer_normalization_31/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/layer_normalization_31/beta/m
?
6Adam/layer_normalization_31/beta/m/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_31/beta/m*
_output_shapes
:@*
dtype0
?
Adam/dense_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_38/kernel/m
?
*Adam/dense_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_38/bias/m
y
(Adam/dense_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/m*
_output_shapes
:*
dtype0
?
#Adam/layer_normalization_34/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/layer_normalization_34/gamma/m
?
7Adam/layer_normalization_34/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_34/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/layer_normalization_34/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/layer_normalization_34/beta/m
?
6Adam/layer_normalization_34/beta/m/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_34/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_39/kernel/m
?
*Adam/dense_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_39/bias/m
y
(Adam/dense_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/m*
_output_shapes
:*
dtype0
?
CAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*T
shared_nameECAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/m
?
WAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/m/Read/ReadVariableOpReadVariableOpCAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/m*
_output_shapes

:@*
dtype0
?
AAdam/transformer_block_9/multi_head_self_attention_9/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAAdam/transformer_block_9/multi_head_self_attention_9/query/bias/m
?
UAdam/transformer_block_9/multi_head_self_attention_9/query/bias/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_9/multi_head_self_attention_9/query/bias/m*
_output_shapes
:*
dtype0
?
AAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*R
shared_nameCAAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/m
?
UAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/m*
_output_shapes

:@*
dtype0
?
?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/m
?
SAdam/transformer_block_9/multi_head_self_attention_9/key/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/m*
_output_shapes
:*
dtype0
?
CAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*T
shared_nameECAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/m
?
WAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/m/Read/ReadVariableOpReadVariableOpCAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/m*
_output_shapes

:@*
dtype0
?
AAdam/transformer_block_9/multi_head_self_attention_9/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAAdam/transformer_block_9/multi_head_self_attention_9/value/bias/m
?
UAdam/transformer_block_9/multi_head_self_attention_9/value/bias/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_9/multi_head_self_attention_9/value/bias/m*
_output_shapes
:*
dtype0
?
AAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/m
?
UAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/m
?
SAdam/transformer_block_9/multi_head_self_attention_9/out/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_36/kernel/m
?
*Adam/dense_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_36/bias/m
y
(Adam/dense_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_37/kernel/m
?
*Adam/dense_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_37/bias/m
y
(Adam/dense_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/m*
_output_shapes
:*
dtype0
?
7Adam/transformer_block_9/layer_normalization_32/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_block_9/layer_normalization_32/gamma/m
?
KAdam/transformer_block_9/layer_normalization_32/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_9/layer_normalization_32/gamma/m*
_output_shapes
:*
dtype0
?
6Adam/transformer_block_9/layer_normalization_32/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_9/layer_normalization_32/beta/m
?
JAdam/transformer_block_9/layer_normalization_32/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_9/layer_normalization_32/beta/m*
_output_shapes
:*
dtype0
?
7Adam/transformer_block_9/layer_normalization_33/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_block_9/layer_normalization_33/gamma/m
?
KAdam/transformer_block_9/layer_normalization_33/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_9/layer_normalization_33/gamma/m*
_output_shapes
:*
dtype0
?
6Adam/transformer_block_9/layer_normalization_33/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_9/layer_normalization_33/beta/m
?
JAdam/transformer_block_9/layer_normalization_33/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_9/layer_normalization_33/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*'
shared_nameAdam/dense_35/kernel/v
?
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes

:d@*
dtype0
?
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:@*
dtype0
?
#Adam/layer_normalization_31/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/layer_normalization_31/gamma/v
?
7Adam/layer_normalization_31/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_31/gamma/v*
_output_shapes
:@*
dtype0
?
"Adam/layer_normalization_31/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/layer_normalization_31/beta/v
?
6Adam/layer_normalization_31/beta/v/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_31/beta/v*
_output_shapes
:@*
dtype0
?
Adam/dense_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_38/kernel/v
?
*Adam/dense_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_38/bias/v
y
(Adam/dense_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/v*
_output_shapes
:*
dtype0
?
#Adam/layer_normalization_34/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/layer_normalization_34/gamma/v
?
7Adam/layer_normalization_34/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_34/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/layer_normalization_34/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/layer_normalization_34/beta/v
?
6Adam/layer_normalization_34/beta/v/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_34/beta/v*
_output_shapes
:*
dtype0
?
Adam/dense_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_39/kernel/v
?
*Adam/dense_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_39/bias/v
y
(Adam/dense_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/v*
_output_shapes
:*
dtype0
?
CAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*T
shared_nameECAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/v
?
WAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/v/Read/ReadVariableOpReadVariableOpCAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/v*
_output_shapes

:@*
dtype0
?
AAdam/transformer_block_9/multi_head_self_attention_9/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAAdam/transformer_block_9/multi_head_self_attention_9/query/bias/v
?
UAdam/transformer_block_9/multi_head_self_attention_9/query/bias/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_9/multi_head_self_attention_9/query/bias/v*
_output_shapes
:*
dtype0
?
AAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*R
shared_nameCAAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/v
?
UAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/v*
_output_shapes

:@*
dtype0
?
?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/v
?
SAdam/transformer_block_9/multi_head_self_attention_9/key/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/v*
_output_shapes
:*
dtype0
?
CAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*T
shared_nameECAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/v
?
WAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/v/Read/ReadVariableOpReadVariableOpCAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/v*
_output_shapes

:@*
dtype0
?
AAdam/transformer_block_9/multi_head_self_attention_9/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAAdam/transformer_block_9/multi_head_self_attention_9/value/bias/v
?
UAdam/transformer_block_9/multi_head_self_attention_9/value/bias/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_9/multi_head_self_attention_9/value/bias/v*
_output_shapes
:*
dtype0
?
AAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/v
?
UAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/v
?
SAdam/transformer_block_9/multi_head_self_attention_9/out/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_36/kernel/v
?
*Adam/dense_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_36/bias/v
y
(Adam/dense_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_37/kernel/v
?
*Adam/dense_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_37/bias/v
y
(Adam/dense_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/v*
_output_shapes
:*
dtype0
?
7Adam/transformer_block_9/layer_normalization_32/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_block_9/layer_normalization_32/gamma/v
?
KAdam/transformer_block_9/layer_normalization_32/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_9/layer_normalization_32/gamma/v*
_output_shapes
:*
dtype0
?
6Adam/transformer_block_9/layer_normalization_32/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_9/layer_normalization_32/beta/v
?
JAdam/transformer_block_9/layer_normalization_32/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_9/layer_normalization_32/beta/v*
_output_shapes
:*
dtype0
?
7Adam/transformer_block_9/layer_normalization_33/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_block_9/layer_normalization_33/gamma/v
?
KAdam/transformer_block_9/layer_normalization_33/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_9/layer_normalization_33/gamma/v*
_output_shapes
:*
dtype0
?
6Adam/transformer_block_9/layer_normalization_33/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_9/layer_normalization_33/beta/v
?
JAdam/transformer_block_9/layer_normalization_33/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_9/layer_normalization_33/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
	variables
trainable_variables
regularization_losses
	keras_api
 _random_generator
!__call__
*"&call_and_return_all_conditional_losses* 
?
#axis
	$gamma
%beta
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
?
,att
-ffn
.
layernorm1
/
layernorm2
0dropout1
1dropout2
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses*
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
?
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B_random_generator
C__call__
*D&call_and_return_all_conditional_losses* 
?

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses*
?
Maxis
	Ngamma
Obeta
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses*
?

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses*
?
^iter

_beta_1

`beta_2
	adecay
blearning_ratem?m?$m?%m?Em?Fm?Nm?Om?Vm?Wm?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?om?pm?qm?rm?v?v?$v?%v?Ev?Fv?Nv?Ov?Vv?Wv?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?*
?
0
1
$2
%3
c4
d5
e6
f7
g8
h9
i10
j11
k12
l13
m14
n15
o16
p17
q18
r19
E20
F21
N22
O23
V24
W25*
?
0
1
$2
%3
c4
d5
e6
f7
g8
h9
i10
j11
k12
l13
m14
n15
o16
p17
q18
r19
E20
F21
N22
O23
V24
W25*

s0
t1
u2* 
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

{serving_default* 
_Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_35/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
	
s0* 
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 
* 
* 
* 
* 
ke
VARIABLE_VALUElayer_normalization_31/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_31/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
?
?query_dense
?	key_dense
?value_dense
?combine_heads
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis
	ogamma
pbeta
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis
	qgamma
rbeta
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses* 
z
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15*
z
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
>	variables
?trainable_variables
@regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_38/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_38/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

E0
F1*

E0
F1*
	
t0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUElayer_normalization_34/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_34/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*

N0
O1*

N0
O1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_39/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_39/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

V0
W1*

V0
W1*
	
u0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<transformer_block_9/multi_head_self_attention_9/query/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE:transformer_block_9/multi_head_self_attention_9/query/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE:transformer_block_9/multi_head_self_attention_9/key/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8transformer_block_9/multi_head_self_attention_9/key/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<transformer_block_9/multi_head_self_attention_9/value/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE:transformer_block_9/multi_head_self_attention_9/value/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_block_9/multi_head_self_attention_9/out/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_block_9/multi_head_self_attention_9/out/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_36/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_36/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_37/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_37/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0transformer_block_9/layer_normalization_32/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_block_9/layer_normalization_32/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0transformer_block_9/layer_normalization_33/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_block_9/layer_normalization_33/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
J
0
1
2
3
4
5
6
7
	8

9*

?0
?1*
* 
* 
* 
* 
* 
* 
	
s0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?

ckernel
dbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

ekernel
fbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

gkernel
hbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

ikernel
jbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
<
c0
d1
e2
f3
g4
h5
i6
j7*
<
c0
d1
e2
f3
g4
h5
i6
j7*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?

kkernel
lbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

mkernel
nbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
 
k0
l1
m2
n3*
 
k0
l1
m2
n3*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 

o0
p1*

o0
p1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 

q0
r1*

q0
r1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
.
,0
-1
.2
/3
04
15*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
t0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
u0* 
* 
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*

c0
d1*

c0
d1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

e0
f1*

e0
f1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

g0
h1*

g0
h1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

i0
j1*

i0
j1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
$
?0
?1
?2
?3*
* 
* 
* 

k0
l1*

k0
l1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

m0
n1*

m0
n1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?|
VARIABLE_VALUEAdam/dense_35/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_35/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/layer_normalization_31/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/layer_normalization_31/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_38/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_38/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/layer_normalization_34/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/layer_normalization_34/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_39/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_39/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAAdam/transformer_block_9/multi_head_self_attention_9/query/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAAdam/transformer_block_9/multi_head_self_attention_9/value/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_36/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_36/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_37/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_37/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_block_9/layer_normalization_32/gamma/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_block_9/layer_normalization_32/beta/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_block_9/layer_normalization_33/gamma/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_block_9/layer_normalization_33/beta/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_35/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_35/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/layer_normalization_31/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/layer_normalization_31/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_38/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_38/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/layer_normalization_34/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/layer_normalization_34/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_39/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_39/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAAdam/transformer_block_9/multi_head_self_attention_9/query/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAAdam/transformer_block_9/multi_head_self_attention_9/value/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_36/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_36/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_37/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_37/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_block_9/layer_normalization_32/gamma/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_block_9/layer_normalization_32/beta/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_block_9/layer_normalization_33/gamma/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_block_9/layer_normalization_33/beta/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_10Placeholder*+
_output_shapes
:?????????pd*
dtype0* 
shape:?????????pd
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10dense_35/kerneldense_35/biaslayer_normalization_31/gammalayer_normalization_31/beta<transformer_block_9/multi_head_self_attention_9/query/kernel:transformer_block_9/multi_head_self_attention_9/query/bias:transformer_block_9/multi_head_self_attention_9/key/kernel8transformer_block_9/multi_head_self_attention_9/key/bias<transformer_block_9/multi_head_self_attention_9/value/kernel:transformer_block_9/multi_head_self_attention_9/value/bias:transformer_block_9/multi_head_self_attention_9/out/kernel8transformer_block_9/multi_head_self_attention_9/out/bias0transformer_block_9/layer_normalization_32/gamma/transformer_block_9/layer_normalization_32/betadense_36/kerneldense_36/biasdense_37/kerneldense_37/bias0transformer_block_9/layer_normalization_33/gamma/transformer_block_9/layer_normalization_33/betadense_38/kerneldense_38/biaslayer_normalization_34/gammalayer_normalization_34/betadense_39/kerneldense_39/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_258490
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?*
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOp0layer_normalization_31/gamma/Read/ReadVariableOp/layer_normalization_31/beta/Read/ReadVariableOp#dense_38/kernel/Read/ReadVariableOp!dense_38/bias/Read/ReadVariableOp0layer_normalization_34/gamma/Read/ReadVariableOp/layer_normalization_34/beta/Read/ReadVariableOp#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpPtransformer_block_9/multi_head_self_attention_9/query/kernel/Read/ReadVariableOpNtransformer_block_9/multi_head_self_attention_9/query/bias/Read/ReadVariableOpNtransformer_block_9/multi_head_self_attention_9/key/kernel/Read/ReadVariableOpLtransformer_block_9/multi_head_self_attention_9/key/bias/Read/ReadVariableOpPtransformer_block_9/multi_head_self_attention_9/value/kernel/Read/ReadVariableOpNtransformer_block_9/multi_head_self_attention_9/value/bias/Read/ReadVariableOpNtransformer_block_9/multi_head_self_attention_9/out/kernel/Read/ReadVariableOpLtransformer_block_9/multi_head_self_attention_9/out/bias/Read/ReadVariableOp#dense_36/kernel/Read/ReadVariableOp!dense_36/bias/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOpDtransformer_block_9/layer_normalization_32/gamma/Read/ReadVariableOpCtransformer_block_9/layer_normalization_32/beta/Read/ReadVariableOpDtransformer_block_9/layer_normalization_33/gamma/Read/ReadVariableOpCtransformer_block_9/layer_normalization_33/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp7Adam/layer_normalization_31/gamma/m/Read/ReadVariableOp6Adam/layer_normalization_31/beta/m/Read/ReadVariableOp*Adam/dense_38/kernel/m/Read/ReadVariableOp(Adam/dense_38/bias/m/Read/ReadVariableOp7Adam/layer_normalization_34/gamma/m/Read/ReadVariableOp6Adam/layer_normalization_34/beta/m/Read/ReadVariableOp*Adam/dense_39/kernel/m/Read/ReadVariableOp(Adam/dense_39/bias/m/Read/ReadVariableOpWAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/m/Read/ReadVariableOpUAdam/transformer_block_9/multi_head_self_attention_9/query/bias/m/Read/ReadVariableOpUAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/m/Read/ReadVariableOpSAdam/transformer_block_9/multi_head_self_attention_9/key/bias/m/Read/ReadVariableOpWAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/m/Read/ReadVariableOpUAdam/transformer_block_9/multi_head_self_attention_9/value/bias/m/Read/ReadVariableOpUAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/m/Read/ReadVariableOpSAdam/transformer_block_9/multi_head_self_attention_9/out/bias/m/Read/ReadVariableOp*Adam/dense_36/kernel/m/Read/ReadVariableOp(Adam/dense_36/bias/m/Read/ReadVariableOp*Adam/dense_37/kernel/m/Read/ReadVariableOp(Adam/dense_37/bias/m/Read/ReadVariableOpKAdam/transformer_block_9/layer_normalization_32/gamma/m/Read/ReadVariableOpJAdam/transformer_block_9/layer_normalization_32/beta/m/Read/ReadVariableOpKAdam/transformer_block_9/layer_normalization_33/gamma/m/Read/ReadVariableOpJAdam/transformer_block_9/layer_normalization_33/beta/m/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOp7Adam/layer_normalization_31/gamma/v/Read/ReadVariableOp6Adam/layer_normalization_31/beta/v/Read/ReadVariableOp*Adam/dense_38/kernel/v/Read/ReadVariableOp(Adam/dense_38/bias/v/Read/ReadVariableOp7Adam/layer_normalization_34/gamma/v/Read/ReadVariableOp6Adam/layer_normalization_34/beta/v/Read/ReadVariableOp*Adam/dense_39/kernel/v/Read/ReadVariableOp(Adam/dense_39/bias/v/Read/ReadVariableOpWAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/v/Read/ReadVariableOpUAdam/transformer_block_9/multi_head_self_attention_9/query/bias/v/Read/ReadVariableOpUAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/v/Read/ReadVariableOpSAdam/transformer_block_9/multi_head_self_attention_9/key/bias/v/Read/ReadVariableOpWAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/v/Read/ReadVariableOpUAdam/transformer_block_9/multi_head_self_attention_9/value/bias/v/Read/ReadVariableOpUAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/v/Read/ReadVariableOpSAdam/transformer_block_9/multi_head_self_attention_9/out/bias/v/Read/ReadVariableOp*Adam/dense_36/kernel/v/Read/ReadVariableOp(Adam/dense_36/bias/v/Read/ReadVariableOp*Adam/dense_37/kernel/v/Read/ReadVariableOp(Adam/dense_37/bias/v/Read/ReadVariableOpKAdam/transformer_block_9/layer_normalization_32/gamma/v/Read/ReadVariableOpJAdam/transformer_block_9/layer_normalization_32/beta/v/Read/ReadVariableOpKAdam/transformer_block_9/layer_normalization_33/gamma/v/Read/ReadVariableOpJAdam/transformer_block_9/layer_normalization_33/beta/v/Read/ReadVariableOpConst*d
Tin]
[2Y	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_259888
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_35/kerneldense_35/biaslayer_normalization_31/gammalayer_normalization_31/betadense_38/kerneldense_38/biaslayer_normalization_34/gammalayer_normalization_34/betadense_39/kerneldense_39/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate<transformer_block_9/multi_head_self_attention_9/query/kernel:transformer_block_9/multi_head_self_attention_9/query/bias:transformer_block_9/multi_head_self_attention_9/key/kernel8transformer_block_9/multi_head_self_attention_9/key/bias<transformer_block_9/multi_head_self_attention_9/value/kernel:transformer_block_9/multi_head_self_attention_9/value/bias:transformer_block_9/multi_head_self_attention_9/out/kernel8transformer_block_9/multi_head_self_attention_9/out/biasdense_36/kerneldense_36/biasdense_37/kerneldense_37/bias0transformer_block_9/layer_normalization_32/gamma/transformer_block_9/layer_normalization_32/beta0transformer_block_9/layer_normalization_33/gamma/transformer_block_9/layer_normalization_33/betatotalcounttotal_1count_1Adam/dense_35/kernel/mAdam/dense_35/bias/m#Adam/layer_normalization_31/gamma/m"Adam/layer_normalization_31/beta/mAdam/dense_38/kernel/mAdam/dense_38/bias/m#Adam/layer_normalization_34/gamma/m"Adam/layer_normalization_34/beta/mAdam/dense_39/kernel/mAdam/dense_39/bias/mCAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/mAAdam/transformer_block_9/multi_head_self_attention_9/query/bias/mAAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/m?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/mCAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/mAAdam/transformer_block_9/multi_head_self_attention_9/value/bias/mAAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/m?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/mAdam/dense_36/kernel/mAdam/dense_36/bias/mAdam/dense_37/kernel/mAdam/dense_37/bias/m7Adam/transformer_block_9/layer_normalization_32/gamma/m6Adam/transformer_block_9/layer_normalization_32/beta/m7Adam/transformer_block_9/layer_normalization_33/gamma/m6Adam/transformer_block_9/layer_normalization_33/beta/mAdam/dense_35/kernel/vAdam/dense_35/bias/v#Adam/layer_normalization_31/gamma/v"Adam/layer_normalization_31/beta/vAdam/dense_38/kernel/vAdam/dense_38/bias/v#Adam/layer_normalization_34/gamma/v"Adam/layer_normalization_34/beta/vAdam/dense_39/kernel/vAdam/dense_39/bias/vCAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/vAAdam/transformer_block_9/multi_head_self_attention_9/query/bias/vAAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/v?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/vCAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/vAAdam/transformer_block_9/multi_head_self_attention_9/value/bias/vAAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/v?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/vAdam/dense_36/kernel/vAdam/dense_36/bias/vAdam/dense_37/kernel/vAdam/dense_37/bias/v7Adam/transformer_block_9/layer_normalization_32/gamma/v6Adam/transformer_block_9/layer_normalization_32/beta/v7Adam/transformer_block_9/layer_normalization_33/gamma/v6Adam/transformer_block_9/layer_normalization_33/beta/v*c
Tin\
Z2X*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_260159??(
?
?
(__inference_model_4_layer_call_fn_257325
input_10
unknown:d@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:	?

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_257213o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????pd
"
_user_specified_name
input_10
??
?
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_259199

inputsU
Cmulti_head_self_attention_9_query_tensordot_readvariableop_resource:@O
Amulti_head_self_attention_9_query_biasadd_readvariableop_resource:S
Amulti_head_self_attention_9_key_tensordot_readvariableop_resource:@M
?multi_head_self_attention_9_key_biasadd_readvariableop_resource:U
Cmulti_head_self_attention_9_value_tensordot_readvariableop_resource:@O
Amulti_head_self_attention_9_value_biasadd_readvariableop_resource:S
Amulti_head_self_attention_9_out_tensordot_readvariableop_resource:M
?multi_head_self_attention_9_out_biasadd_readvariableop_resource:J
<layer_normalization_32_batchnorm_mul_readvariableop_resource:F
8layer_normalization_32_batchnorm_readvariableop_resource:I
7sequential_9_dense_36_tensordot_readvariableop_resource:C
5sequential_9_dense_36_biasadd_readvariableop_resource:I
7sequential_9_dense_37_tensordot_readvariableop_resource:C
5sequential_9_dense_37_biasadd_readvariableop_resource:J
<layer_normalization_33_batchnorm_mul_readvariableop_resource:F
8layer_normalization_33_batchnorm_readvariableop_resource:
identity??/layer_normalization_32/batchnorm/ReadVariableOp?3layer_normalization_32/batchnorm/mul/ReadVariableOp?/layer_normalization_33/batchnorm/ReadVariableOp?3layer_normalization_33/batchnorm/mul/ReadVariableOp?6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp?8multi_head_self_attention_9/key/Tensordot/ReadVariableOp?6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp?8multi_head_self_attention_9/out/Tensordot/ReadVariableOp?8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp?:multi_head_self_attention_9/query/Tensordot/ReadVariableOp?8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp?:multi_head_self_attention_9/value/Tensordot/ReadVariableOp?,sequential_9/dense_36/BiasAdd/ReadVariableOp?.sequential_9/dense_36/Tensordot/ReadVariableOp?,sequential_9/dense_37/BiasAdd/ReadVariableOp?.sequential_9/dense_37/Tensordot/ReadVariableOpW
!multi_head_self_attention_9/ShapeShapeinputs*
T0*
_output_shapes
:y
/multi_head_self_attention_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1multi_head_self_attention_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1multi_head_self_attention_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)multi_head_self_attention_9/strided_sliceStridedSlice*multi_head_self_attention_9/Shape:output:08multi_head_self_attention_9/strided_slice/stack:output:0:multi_head_self_attention_9/strided_slice/stack_1:output:0:multi_head_self_attention_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:multi_head_self_attention_9/query/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_9_query_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0z
0multi_head_self_attention_9/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0multi_head_self_attention_9/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
1multi_head_self_attention_9/query/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention_9/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/query/Tensordot/GatherV2GatherV2:multi_head_self_attention_9/query/Tensordot/Shape:output:09multi_head_self_attention_9/query/Tensordot/free:output:0Bmulti_head_self_attention_9/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;multi_head_self_attention_9/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6multi_head_self_attention_9/query/Tensordot/GatherV2_1GatherV2:multi_head_self_attention_9/query/Tensordot/Shape:output:09multi_head_self_attention_9/query/Tensordot/axes:output:0Dmulti_head_self_attention_9/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1multi_head_self_attention_9/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/query/Tensordot/ProdProd=multi_head_self_attention_9/query/Tensordot/GatherV2:output:0:multi_head_self_attention_9/query/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention_9/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2multi_head_self_attention_9/query/Tensordot/Prod_1Prod?multi_head_self_attention_9/query/Tensordot/GatherV2_1:output:0<multi_head_self_attention_9/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention_9/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/query/Tensordot/concatConcatV29multi_head_self_attention_9/query/Tensordot/free:output:09multi_head_self_attention_9/query/Tensordot/axes:output:0@multi_head_self_attention_9/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1multi_head_self_attention_9/query/Tensordot/stackPack9multi_head_self_attention_9/query/Tensordot/Prod:output:0;multi_head_self_attention_9/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5multi_head_self_attention_9/query/Tensordot/transpose	Transposeinputs;multi_head_self_attention_9/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
3multi_head_self_attention_9/query/Tensordot/ReshapeReshape9multi_head_self_attention_9/query/Tensordot/transpose:y:0:multi_head_self_attention_9/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2multi_head_self_attention_9/query/Tensordot/MatMulMatMul<multi_head_self_attention_9/query/Tensordot/Reshape:output:0Bmulti_head_self_attention_9/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
3multi_head_self_attention_9/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:{
9multi_head_self_attention_9/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/query/Tensordot/concat_1ConcatV2=multi_head_self_attention_9/query/Tensordot/GatherV2:output:0<multi_head_self_attention_9/query/Tensordot/Const_2:output:0Bmulti_head_self_attention_9/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+multi_head_self_attention_9/query/TensordotReshape<multi_head_self_attention_9/query/Tensordot/MatMul:product:0=multi_head_self_attention_9/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/query/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_query_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_self_attention_9/query/BiasAddBiasAdd4multi_head_self_attention_9/query/Tensordot:output:0@multi_head_self_attention_9/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/key/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_key_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0x
.multi_head_self_attention_9/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention_9/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_self_attention_9/key/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_self_attention_9/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/key/Tensordot/GatherV2GatherV28multi_head_self_attention_9/key/Tensordot/Shape:output:07multi_head_self_attention_9/key/Tensordot/free:output:0@multi_head_self_attention_9/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention_9/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/key/Tensordot/GatherV2_1GatherV28multi_head_self_attention_9/key/Tensordot/Shape:output:07multi_head_self_attention_9/key/Tensordot/axes:output:0Bmulti_head_self_attention_9/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention_9/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_self_attention_9/key/Tensordot/ProdProd;multi_head_self_attention_9/key/Tensordot/GatherV2:output:08multi_head_self_attention_9/key/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention_9/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/key/Tensordot/Prod_1Prod=multi_head_self_attention_9/key/Tensordot/GatherV2_1:output:0:multi_head_self_attention_9/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention_9/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_self_attention_9/key/Tensordot/concatConcatV27multi_head_self_attention_9/key/Tensordot/free:output:07multi_head_self_attention_9/key/Tensordot/axes:output:0>multi_head_self_attention_9/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_self_attention_9/key/Tensordot/stackPack7multi_head_self_attention_9/key/Tensordot/Prod:output:09multi_head_self_attention_9/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_self_attention_9/key/Tensordot/transpose	Transposeinputs9multi_head_self_attention_9/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
1multi_head_self_attention_9/key/Tensordot/ReshapeReshape7multi_head_self_attention_9/key/Tensordot/transpose:y:08multi_head_self_attention_9/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_self_attention_9/key/Tensordot/MatMulMatMul:multi_head_self_attention_9/key/Tensordot/Reshape:output:0@multi_head_self_attention_9/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_self_attention_9/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_self_attention_9/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/key/Tensordot/concat_1ConcatV2;multi_head_self_attention_9/key/Tensordot/GatherV2:output:0:multi_head_self_attention_9/key/Tensordot/Const_2:output:0@multi_head_self_attention_9/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_self_attention_9/key/TensordotReshape:multi_head_self_attention_9/key/Tensordot/MatMul:product:0;multi_head_self_attention_9/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
6multi_head_self_attention_9/key/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_9_key_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_self_attention_9/key/BiasAddBiasAdd2multi_head_self_attention_9/key/Tensordot:output:0>multi_head_self_attention_9/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
:multi_head_self_attention_9/value/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_9_value_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0z
0multi_head_self_attention_9/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0multi_head_self_attention_9/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
1multi_head_self_attention_9/value/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention_9/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/value/Tensordot/GatherV2GatherV2:multi_head_self_attention_9/value/Tensordot/Shape:output:09multi_head_self_attention_9/value/Tensordot/free:output:0Bmulti_head_self_attention_9/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;multi_head_self_attention_9/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6multi_head_self_attention_9/value/Tensordot/GatherV2_1GatherV2:multi_head_self_attention_9/value/Tensordot/Shape:output:09multi_head_self_attention_9/value/Tensordot/axes:output:0Dmulti_head_self_attention_9/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1multi_head_self_attention_9/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/value/Tensordot/ProdProd=multi_head_self_attention_9/value/Tensordot/GatherV2:output:0:multi_head_self_attention_9/value/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention_9/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2multi_head_self_attention_9/value/Tensordot/Prod_1Prod?multi_head_self_attention_9/value/Tensordot/GatherV2_1:output:0<multi_head_self_attention_9/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention_9/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/value/Tensordot/concatConcatV29multi_head_self_attention_9/value/Tensordot/free:output:09multi_head_self_attention_9/value/Tensordot/axes:output:0@multi_head_self_attention_9/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1multi_head_self_attention_9/value/Tensordot/stackPack9multi_head_self_attention_9/value/Tensordot/Prod:output:0;multi_head_self_attention_9/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5multi_head_self_attention_9/value/Tensordot/transpose	Transposeinputs;multi_head_self_attention_9/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
3multi_head_self_attention_9/value/Tensordot/ReshapeReshape9multi_head_self_attention_9/value/Tensordot/transpose:y:0:multi_head_self_attention_9/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2multi_head_self_attention_9/value/Tensordot/MatMulMatMul<multi_head_self_attention_9/value/Tensordot/Reshape:output:0Bmulti_head_self_attention_9/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
3multi_head_self_attention_9/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:{
9multi_head_self_attention_9/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/value/Tensordot/concat_1ConcatV2=multi_head_self_attention_9/value/Tensordot/GatherV2:output:0<multi_head_self_attention_9/value/Tensordot/Const_2:output:0Bmulti_head_self_attention_9/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+multi_head_self_attention_9/value/TensordotReshape<multi_head_self_attention_9/value/Tensordot/MatMul:product:0=multi_head_self_attention_9/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/value/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_self_attention_9/value/BiasAddBiasAdd4multi_head_self_attention_9/value/Tensordot:output:0@multi_head_self_attention_9/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pm
+multi_head_self_attention_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :pm
+multi_head_self_attention_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention_9/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
)multi_head_self_attention_9/Reshape/shapePack2multi_head_self_attention_9/strided_slice:output:04multi_head_self_attention_9/Reshape/shape/1:output:04multi_head_self_attention_9/Reshape/shape/2:output:04multi_head_self_attention_9/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
#multi_head_self_attention_9/ReshapeReshape2multi_head_self_attention_9/query/BiasAdd:output:02multi_head_self_attention_9/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????p?
*multi_head_self_attention_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
%multi_head_self_attention_9/transpose	Transpose,multi_head_self_attention_9/Reshape:output:03multi_head_self_attention_9/transpose/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-multi_head_self_attention_9/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_1/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_1/shape/1:output:06multi_head_self_attention_9/Reshape_1/shape/2:output:06multi_head_self_attention_9/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_1Reshape0multi_head_self_attention_9/key/BiasAdd:output:04multi_head_self_attention_9/Reshape_1/shape:output:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_1	Transpose.multi_head_self_attention_9/Reshape_1:output:05multi_head_self_attention_9/transpose_1/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-multi_head_self_attention_9/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_2/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_2/shape/1:output:06multi_head_self_attention_9/Reshape_2/shape/2:output:06multi_head_self_attention_9/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_2Reshape2multi_head_self_attention_9/value/BiasAdd:output:04multi_head_self_attention_9/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_2	Transpose.multi_head_self_attention_9/Reshape_2:output:05multi_head_self_attention_9/transpose_2/perm:output:0*
T0*/
_output_shapes
:?????????p?
"multi_head_self_attention_9/MatMulBatchMatMulV2)multi_head_self_attention_9/transpose:y:0+multi_head_self_attention_9/transpose_1:y:0*
T0*/
_output_shapes
:?????????pp*
adj_y(~
#multi_head_self_attention_9/Shape_1Shape+multi_head_self_attention_9/transpose_1:y:0*
T0*
_output_shapes
:?
1multi_head_self_attention_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
3multi_head_self_attention_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3multi_head_self_attention_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+multi_head_self_attention_9/strided_slice_1StridedSlice,multi_head_self_attention_9/Shape_1:output:0:multi_head_self_attention_9/strided_slice_1/stack:output:0<multi_head_self_attention_9/strided_slice_1/stack_1:output:0<multi_head_self_attention_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
 multi_head_self_attention_9/CastCast4multi_head_self_attention_9/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: o
 multi_head_self_attention_9/SqrtSqrt$multi_head_self_attention_9/Cast:y:0*
T0*
_output_shapes
: ?
#multi_head_self_attention_9/truedivRealDiv+multi_head_self_attention_9/MatMul:output:0$multi_head_self_attention_9/Sqrt:y:0*
T0*/
_output_shapes
:?????????pp?
#multi_head_self_attention_9/SoftmaxSoftmax'multi_head_self_attention_9/truediv:z:0*
T0*/
_output_shapes
:?????????pp?
$multi_head_self_attention_9/MatMul_1BatchMatMulV2-multi_head_self_attention_9/Softmax:softmax:0+multi_head_self_attention_9/transpose_2:y:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_3	Transpose-multi_head_self_attention_9/MatMul_1:output:05multi_head_self_attention_9/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_3/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_3/shape/1:output:06multi_head_self_attention_9/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_3Reshape+multi_head_self_attention_9/transpose_3:y:04multi_head_self_attention_9/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/out/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_self_attention_9/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention_9/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
/multi_head_self_attention_9/out/Tensordot/ShapeShape.multi_head_self_attention_9/Reshape_3:output:0*
T0*
_output_shapes
:y
7multi_head_self_attention_9/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/out/Tensordot/GatherV2GatherV28multi_head_self_attention_9/out/Tensordot/Shape:output:07multi_head_self_attention_9/out/Tensordot/free:output:0@multi_head_self_attention_9/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention_9/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/out/Tensordot/GatherV2_1GatherV28multi_head_self_attention_9/out/Tensordot/Shape:output:07multi_head_self_attention_9/out/Tensordot/axes:output:0Bmulti_head_self_attention_9/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention_9/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_self_attention_9/out/Tensordot/ProdProd;multi_head_self_attention_9/out/Tensordot/GatherV2:output:08multi_head_self_attention_9/out/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention_9/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/out/Tensordot/Prod_1Prod=multi_head_self_attention_9/out/Tensordot/GatherV2_1:output:0:multi_head_self_attention_9/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention_9/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_self_attention_9/out/Tensordot/concatConcatV27multi_head_self_attention_9/out/Tensordot/free:output:07multi_head_self_attention_9/out/Tensordot/axes:output:0>multi_head_self_attention_9/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_self_attention_9/out/Tensordot/stackPack7multi_head_self_attention_9/out/Tensordot/Prod:output:09multi_head_self_attention_9/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_self_attention_9/out/Tensordot/transpose	Transpose.multi_head_self_attention_9/Reshape_3:output:09multi_head_self_attention_9/out/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
1multi_head_self_attention_9/out/Tensordot/ReshapeReshape7multi_head_self_attention_9/out/Tensordot/transpose:y:08multi_head_self_attention_9/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_self_attention_9/out/Tensordot/MatMulMatMul:multi_head_self_attention_9/out/Tensordot/Reshape:output:0@multi_head_self_attention_9/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_self_attention_9/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_self_attention_9/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/out/Tensordot/concat_1ConcatV2;multi_head_self_attention_9/out/Tensordot/GatherV2:output:0:multi_head_self_attention_9/out/Tensordot/Const_2:output:0@multi_head_self_attention_9/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_self_attention_9/out/TensordotReshape:multi_head_self_attention_9/out/Tensordot/MatMul:product:0;multi_head_self_attention_9/out/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
6multi_head_self_attention_9/out/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_9_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_self_attention_9/out/BiasAddBiasAdd2multi_head_self_attention_9/out/Tensordot:output:0>multi_head_self_attention_9/out/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p]
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_28/dropout/MulMul0multi_head_self_attention_9/out/BiasAdd:output:0!dropout_28/dropout/Const:output:0*
T0*+
_output_shapes
:?????????px
dropout_28/dropout/ShapeShape0multi_head_self_attention_9/out/BiasAdd:output:0*
T0*
_output_shapes
:?
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????p*
dtype0f
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????p?
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????p?
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????p
5layer_normalization_32/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_32/moments/meanMeandropout_28/dropout/Mul_1:z:0>layer_normalization_32/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
+layer_normalization_32/moments/StopGradientStopGradient,layer_normalization_32/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
0layer_normalization_32/moments/SquaredDifferenceSquaredDifferencedropout_28/dropout/Mul_1:z:04layer_normalization_32/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
9layer_normalization_32/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
'layer_normalization_32/moments/varianceMean4layer_normalization_32/moments/SquaredDifference:z:0Blayer_normalization_32/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(k
&layer_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
$layer_normalization_32/batchnorm/addAddV20layer_normalization_32/moments/variance:output:0/layer_normalization_32/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/RsqrtRsqrt(layer_normalization_32/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
3layer_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_32/batchnorm/mulMul*layer_normalization_32/batchnorm/Rsqrt:y:0;layer_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/mul_1Muldropout_28/dropout/Mul_1:z:0(layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/mul_2Mul,layer_normalization_32/moments/mean:output:0(layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
/layer_normalization_32/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_32/batchnorm/subSub7layer_normalization_32/batchnorm/ReadVariableOp:value:0*layer_normalization_32/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/add_1AddV2*layer_normalization_32/batchnorm/mul_1:z:0(layer_normalization_32/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p?
.sequential_9/dense_36/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_36_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_9/dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_9/dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_9/dense_36/Tensordot/ShapeShape*layer_normalization_32/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_9/dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_36/Tensordot/GatherV2GatherV2.sequential_9/dense_36/Tensordot/Shape:output:0-sequential_9/dense_36/Tensordot/free:output:06sequential_9/dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_9/dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_9/dense_36/Tensordot/GatherV2_1GatherV2.sequential_9/dense_36/Tensordot/Shape:output:0-sequential_9/dense_36/Tensordot/axes:output:08sequential_9/dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_9/dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_9/dense_36/Tensordot/ProdProd1sequential_9/dense_36/Tensordot/GatherV2:output:0.sequential_9/dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_9/dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_9/dense_36/Tensordot/Prod_1Prod3sequential_9/dense_36/Tensordot/GatherV2_1:output:00sequential_9/dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_9/dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_9/dense_36/Tensordot/concatConcatV2-sequential_9/dense_36/Tensordot/free:output:0-sequential_9/dense_36/Tensordot/axes:output:04sequential_9/dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_9/dense_36/Tensordot/stackPack-sequential_9/dense_36/Tensordot/Prod:output:0/sequential_9/dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_9/dense_36/Tensordot/transpose	Transpose*layer_normalization_32/batchnorm/add_1:z:0/sequential_9/dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
'sequential_9/dense_36/Tensordot/ReshapeReshape-sequential_9/dense_36/Tensordot/transpose:y:0.sequential_9/dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_9/dense_36/Tensordot/MatMulMatMul0sequential_9/dense_36/Tensordot/Reshape:output:06sequential_9/dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_9/dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_9/dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_36/Tensordot/concat_1ConcatV21sequential_9/dense_36/Tensordot/GatherV2:output:00sequential_9/dense_36/Tensordot/Const_2:output:06sequential_9/dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_9/dense_36/TensordotReshape0sequential_9/dense_36/Tensordot/MatMul:product:01sequential_9/dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/dense_36/BiasAddBiasAdd(sequential_9/dense_36/Tensordot:output:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
.sequential_9/dense_37/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_37_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_9/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_9/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_9/dense_37/Tensordot/ShapeShape(sequential_9/dense_36/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_9/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_37/Tensordot/GatherV2GatherV2.sequential_9/dense_37/Tensordot/Shape:output:0-sequential_9/dense_37/Tensordot/free:output:06sequential_9/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_9/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_9/dense_37/Tensordot/GatherV2_1GatherV2.sequential_9/dense_37/Tensordot/Shape:output:0-sequential_9/dense_37/Tensordot/axes:output:08sequential_9/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_9/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_9/dense_37/Tensordot/ProdProd1sequential_9/dense_37/Tensordot/GatherV2:output:0.sequential_9/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_9/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_9/dense_37/Tensordot/Prod_1Prod3sequential_9/dense_37/Tensordot/GatherV2_1:output:00sequential_9/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_9/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_9/dense_37/Tensordot/concatConcatV2-sequential_9/dense_37/Tensordot/free:output:0-sequential_9/dense_37/Tensordot/axes:output:04sequential_9/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_9/dense_37/Tensordot/stackPack-sequential_9/dense_37/Tensordot/Prod:output:0/sequential_9/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_9/dense_37/Tensordot/transpose	Transpose(sequential_9/dense_36/Relu:activations:0/sequential_9/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
'sequential_9/dense_37/Tensordot/ReshapeReshape-sequential_9/dense_37/Tensordot/transpose:y:0.sequential_9/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_9/dense_37/Tensordot/MatMulMatMul0sequential_9/dense_37/Tensordot/Reshape:output:06sequential_9/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_9/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_9/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_37/Tensordot/concat_1ConcatV21sequential_9/dense_37/Tensordot/GatherV2:output:00sequential_9/dense_37/Tensordot/Const_2:output:06sequential_9/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_9/dense_37/TensordotReshape0sequential_9/dense_37/Tensordot/MatMul:product:01sequential_9/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/dense_37/BiasAddBiasAdd(sequential_9/dense_37/Tensordot:output:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p]
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_29/dropout/MulMul&sequential_9/dense_37/BiasAdd:output:0!dropout_29/dropout/Const:output:0*
T0*+
_output_shapes
:?????????pn
dropout_29/dropout/ShapeShape&sequential_9/dense_37/BiasAdd:output:0*
T0*
_output_shapes
:?
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????p*
dtype0f
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????p?
dropout_29/dropout/CastCast#dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????p?
dropout_29/dropout/Mul_1Muldropout_29/dropout/Mul:z:0dropout_29/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????p?
addAddV2*layer_normalization_32/batchnorm/add_1:z:0dropout_29/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????p
5layer_normalization_33/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_33/moments/meanMeanadd:z:0>layer_normalization_33/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
+layer_normalization_33/moments/StopGradientStopGradient,layer_normalization_33/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
0layer_normalization_33/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_33/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
9layer_normalization_33/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
'layer_normalization_33/moments/varianceMean4layer_normalization_33/moments/SquaredDifference:z:0Blayer_normalization_33/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(k
&layer_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
$layer_normalization_33/batchnorm/addAddV20layer_normalization_33/moments/variance:output:0/layer_normalization_33/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/RsqrtRsqrt(layer_normalization_33/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
3layer_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_33/batchnorm/mulMul*layer_normalization_33/batchnorm/Rsqrt:y:0;layer_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/mul_1Muladd:z:0(layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/mul_2Mul,layer_normalization_33/moments/mean:output:0(layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
/layer_normalization_33/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_33/batchnorm/subSub7layer_normalization_33/batchnorm/ReadVariableOp:value:0*layer_normalization_33/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/add_1AddV2*layer_normalization_33/batchnorm/mul_1:z:0(layer_normalization_33/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p}
IdentityIdentity*layer_normalization_33/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp0^layer_normalization_32/batchnorm/ReadVariableOp4^layer_normalization_32/batchnorm/mul/ReadVariableOp0^layer_normalization_33/batchnorm/ReadVariableOp4^layer_normalization_33/batchnorm/mul/ReadVariableOp7^multi_head_self_attention_9/key/BiasAdd/ReadVariableOp9^multi_head_self_attention_9/key/Tensordot/ReadVariableOp7^multi_head_self_attention_9/out/BiasAdd/ReadVariableOp9^multi_head_self_attention_9/out/Tensordot/ReadVariableOp9^multi_head_self_attention_9/query/BiasAdd/ReadVariableOp;^multi_head_self_attention_9/query/Tensordot/ReadVariableOp9^multi_head_self_attention_9/value/BiasAdd/ReadVariableOp;^multi_head_self_attention_9/value/Tensordot/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp/^sequential_9/dense_36/Tensordot/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp/^sequential_9/dense_37/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????p@: : : : : : : : : : : : : : : : 2b
/layer_normalization_32/batchnorm/ReadVariableOp/layer_normalization_32/batchnorm/ReadVariableOp2j
3layer_normalization_32/batchnorm/mul/ReadVariableOp3layer_normalization_32/batchnorm/mul/ReadVariableOp2b
/layer_normalization_33/batchnorm/ReadVariableOp/layer_normalization_33/batchnorm/ReadVariableOp2j
3layer_normalization_33/batchnorm/mul/ReadVariableOp3layer_normalization_33/batchnorm/mul/ReadVariableOp2p
6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp2t
8multi_head_self_attention_9/key/Tensordot/ReadVariableOp8multi_head_self_attention_9/key/Tensordot/ReadVariableOp2p
6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp2t
8multi_head_self_attention_9/out/Tensordot/ReadVariableOp8multi_head_self_attention_9/out/Tensordot/ReadVariableOp2t
8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp2x
:multi_head_self_attention_9/query/Tensordot/ReadVariableOp:multi_head_self_attention_9/query/Tensordot/ReadVariableOp2t
8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp2x
:multi_head_self_attention_9/value/Tensordot/ReadVariableOp:multi_head_self_attention_9/value/Tensordot/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2`
.sequential_9/dense_36/Tensordot/ReadVariableOp.sequential_9/dense_36/Tensordot/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2`
.sequential_9/dense_37/Tensordot/ReadVariableOp.sequential_9/dense_37/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
?
)__inference_dense_39_layer_call_fn_259335

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_256557o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_258490
input_10
unknown:d@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:	?

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_255874o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????pd
"
_user_specified_name
input_10
?
?
4__inference_transformer_block_9_layer_call_fn_258699

inputs
unknown:@
	unknown_0:
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_256993s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????p@: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?O
?
C__inference_model_4_layer_call_and_return_conditional_losses_257491
input_10!
dense_35_257411:d@
dense_35_257413:@+
layer_normalization_31_257417:@+
layer_normalization_31_257419:@,
transformer_block_9_257422:@(
transformer_block_9_257424:,
transformer_block_9_257426:@(
transformer_block_9_257428:,
transformer_block_9_257430:@(
transformer_block_9_257432:,
transformer_block_9_257434:(
transformer_block_9_257436:(
transformer_block_9_257438:(
transformer_block_9_257440:,
transformer_block_9_257442:(
transformer_block_9_257444:,
transformer_block_9_257446:(
transformer_block_9_257448:(
transformer_block_9_257450:(
transformer_block_9_257452:"
dense_38_257457:	?
dense_38_257459:+
layer_normalization_34_257462:+
layer_normalization_34_257464:!
dense_39_257467:
dense_39_257469:
identity?? dense_35/StatefulPartitionedCall?1dense_35/kernel/Regularizer/Square/ReadVariableOp? dense_38/StatefulPartitionedCall?1dense_38/kernel/Regularizer/Square/ReadVariableOp? dense_39/StatefulPartitionedCall?1dense_39/kernel/Regularizer/Square/ReadVariableOp?"dropout_27/StatefulPartitionedCall?"dropout_30/StatefulPartitionedCall?.layer_normalization_31/StatefulPartitionedCall?.layer_normalization_34/StatefulPartitionedCall?+transformer_block_9/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCallinput_10dense_35_257411dense_35_257413*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_256111?
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_257058?
.layer_normalization_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_27/StatefulPartitionedCall:output:0layer_normalization_31_257417layer_normalization_31_257419*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_31_layer_call_and_return_conditional_losses_256171?
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_31/StatefulPartitionedCall:output:0transformer_block_9_257422transformer_block_9_257424transformer_block_9_257426transformer_block_9_257428transformer_block_9_257430transformer_block_9_257432transformer_block_9_257434transformer_block_9_257436transformer_block_9_257438transformer_block_9_257440transformer_block_9_257442transformer_block_9_257444transformer_block_9_257446transformer_block_9_257448transformer_block_9_257450transformer_block_9_257452*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_256993?
flatten_4/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_256460?
"dropout_30/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0#^dropout_27/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_256687?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall+dropout_30/StatefulPartitionedCall:output:0dense_38_257457dense_38_257459*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_256486?
.layer_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0layer_normalization_34_257462layer_normalization_34_257464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_34_layer_call_and_return_conditional_losses_256534?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_34/StatefulPartitionedCall:output:0dense_39_257467dense_39_257469*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_256557?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_257411*
_output_shapes

:d@*
dtype0?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d@r
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_35/kernel/Regularizer/SumSum&dense_35/kernel/Regularizer/Square:y:0*dense_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38_257457*
_output_shapes
:	?*
dtype0?
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?r
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_39_257467*
_output_shapes

:*
dtype0?
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_35/StatefulPartitionedCall2^dense_35/kernel/Regularizer/Square/ReadVariableOp!^dense_38/StatefulPartitionedCall2^dense_38/kernel/Regularizer/Square/ReadVariableOp!^dense_39/StatefulPartitionedCall2^dense_39/kernel/Regularizer/Square/ReadVariableOp#^dropout_27/StatefulPartitionedCall#^dropout_30/StatefulPartitionedCall/^layer_normalization_31/StatefulPartitionedCall/^layer_normalization_34/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2f
1dense_35/kernel/Regularizer/Square/ReadVariableOp1dense_35/kernel/Regularizer/Square/ReadVariableOp2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2f
1dense_39/kernel/Regularizer/Square/ReadVariableOp1dense_39/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall2H
"dropout_30/StatefulPartitionedCall"dropout_30/StatefulPartitionedCall2`
.layer_normalization_31/StatefulPartitionedCall.layer_normalization_31/StatefulPartitionedCall2`
.layer_normalization_34/StatefulPartitionedCall.layer_normalization_34/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????pd
"
_user_specified_name
input_10
?
F
*__inference_flatten_4_layer_call_fn_259204

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_256460a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????p:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?O
?
C__inference_model_4_layer_call_and_return_conditional_losses_257213

inputs!
dense_35_257133:d@
dense_35_257135:@+
layer_normalization_31_257139:@+
layer_normalization_31_257141:@,
transformer_block_9_257144:@(
transformer_block_9_257146:,
transformer_block_9_257148:@(
transformer_block_9_257150:,
transformer_block_9_257152:@(
transformer_block_9_257154:,
transformer_block_9_257156:(
transformer_block_9_257158:(
transformer_block_9_257160:(
transformer_block_9_257162:,
transformer_block_9_257164:(
transformer_block_9_257166:,
transformer_block_9_257168:(
transformer_block_9_257170:(
transformer_block_9_257172:(
transformer_block_9_257174:"
dense_38_257179:	?
dense_38_257181:+
layer_normalization_34_257184:+
layer_normalization_34_257186:!
dense_39_257189:
dense_39_257191:
identity?? dense_35/StatefulPartitionedCall?1dense_35/kernel/Regularizer/Square/ReadVariableOp? dense_38/StatefulPartitionedCall?1dense_38/kernel/Regularizer/Square/ReadVariableOp? dense_39/StatefulPartitionedCall?1dense_39/kernel/Regularizer/Square/ReadVariableOp?"dropout_27/StatefulPartitionedCall?"dropout_30/StatefulPartitionedCall?.layer_normalization_31/StatefulPartitionedCall?.layer_normalization_34/StatefulPartitionedCall?+transformer_block_9/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCallinputsdense_35_257133dense_35_257135*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_256111?
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_257058?
.layer_normalization_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_27/StatefulPartitionedCall:output:0layer_normalization_31_257139layer_normalization_31_257141*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_31_layer_call_and_return_conditional_losses_256171?
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_31/StatefulPartitionedCall:output:0transformer_block_9_257144transformer_block_9_257146transformer_block_9_257148transformer_block_9_257150transformer_block_9_257152transformer_block_9_257154transformer_block_9_257156transformer_block_9_257158transformer_block_9_257160transformer_block_9_257162transformer_block_9_257164transformer_block_9_257166transformer_block_9_257168transformer_block_9_257170transformer_block_9_257172transformer_block_9_257174*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_256993?
flatten_4/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_256460?
"dropout_30/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0#^dropout_27/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_256687?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall+dropout_30/StatefulPartitionedCall:output:0dense_38_257179dense_38_257181*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_256486?
.layer_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0layer_normalization_34_257184layer_normalization_34_257186*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_34_layer_call_and_return_conditional_losses_256534?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_34/StatefulPartitionedCall:output:0dense_39_257189dense_39_257191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_256557?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_257133*
_output_shapes

:d@*
dtype0?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d@r
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_35/kernel/Regularizer/SumSum&dense_35/kernel/Regularizer/Square:y:0*dense_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38_257179*
_output_shapes
:	?*
dtype0?
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?r
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_39_257189*
_output_shapes

:*
dtype0?
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_35/StatefulPartitionedCall2^dense_35/kernel/Regularizer/Square/ReadVariableOp!^dense_38/StatefulPartitionedCall2^dense_38/kernel/Regularizer/Square/ReadVariableOp!^dense_39/StatefulPartitionedCall2^dense_39/kernel/Regularizer/Square/ReadVariableOp#^dropout_27/StatefulPartitionedCall#^dropout_30/StatefulPartitionedCall/^layer_normalization_31/StatefulPartitionedCall/^layer_normalization_34/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2f
1dense_35/kernel/Regularizer/Square/ReadVariableOp1dense_35/kernel/Regularizer/Square/ReadVariableOp2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2f
1dense_39/kernel/Regularizer/Square/ReadVariableOp1dense_39/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall2H
"dropout_30/StatefulPartitionedCall"dropout_30/StatefulPartitionedCall2`
.layer_normalization_31/StatefulPartitionedCall.layer_normalization_31/StatefulPartitionedCall2`
.layer_normalization_34/StatefulPartitionedCall.layer_normalization_34/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:S O
+
_output_shapes
:?????????pd
 
_user_specified_nameinputs
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_256053
dense_36_input!
dense_36_256042:
dense_36_256044:!
dense_37_256047:
dense_37_256049:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_256042dense_36_256044*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_255912?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_256047dense_37_256049*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_255948|
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????p: : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:[ W
+
_output_shapes
:?????????p
(
_user_specified_namedense_36_input
?	
e
F__inference_dropout_30_layer_call_and_return_conditional_losses_256687

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
e
F__inference_dropout_30_layer_call_and_return_conditional_losses_259237

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?/
__inference__traced_save_259888
file_prefix.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop;
7savev2_layer_normalization_31_gamma_read_readvariableop:
6savev2_layer_normalization_31_beta_read_readvariableop.
*savev2_dense_38_kernel_read_readvariableop,
(savev2_dense_38_bias_read_readvariableop;
7savev2_layer_normalization_34_gamma_read_readvariableop:
6savev2_layer_normalization_34_beta_read_readvariableop.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop[
Wsavev2_transformer_block_9_multi_head_self_attention_9_query_kernel_read_readvariableopY
Usavev2_transformer_block_9_multi_head_self_attention_9_query_bias_read_readvariableopY
Usavev2_transformer_block_9_multi_head_self_attention_9_key_kernel_read_readvariableopW
Ssavev2_transformer_block_9_multi_head_self_attention_9_key_bias_read_readvariableop[
Wsavev2_transformer_block_9_multi_head_self_attention_9_value_kernel_read_readvariableopY
Usavev2_transformer_block_9_multi_head_self_attention_9_value_bias_read_readvariableopY
Usavev2_transformer_block_9_multi_head_self_attention_9_out_kernel_read_readvariableopW
Ssavev2_transformer_block_9_multi_head_self_attention_9_out_bias_read_readvariableop.
*savev2_dense_36_kernel_read_readvariableop,
(savev2_dense_36_bias_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableopO
Ksavev2_transformer_block_9_layer_normalization_32_gamma_read_readvariableopN
Jsavev2_transformer_block_9_layer_normalization_32_beta_read_readvariableopO
Ksavev2_transformer_block_9_layer_normalization_33_gamma_read_readvariableopN
Jsavev2_transformer_block_9_layer_normalization_33_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableopB
>savev2_adam_layer_normalization_31_gamma_m_read_readvariableopA
=savev2_adam_layer_normalization_31_beta_m_read_readvariableop5
1savev2_adam_dense_38_kernel_m_read_readvariableop3
/savev2_adam_dense_38_bias_m_read_readvariableopB
>savev2_adam_layer_normalization_34_gamma_m_read_readvariableopA
=savev2_adam_layer_normalization_34_beta_m_read_readvariableop5
1savev2_adam_dense_39_kernel_m_read_readvariableop3
/savev2_adam_dense_39_bias_m_read_readvariableopb
^savev2_adam_transformer_block_9_multi_head_self_attention_9_query_kernel_m_read_readvariableop`
\savev2_adam_transformer_block_9_multi_head_self_attention_9_query_bias_m_read_readvariableop`
\savev2_adam_transformer_block_9_multi_head_self_attention_9_key_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_9_multi_head_self_attention_9_key_bias_m_read_readvariableopb
^savev2_adam_transformer_block_9_multi_head_self_attention_9_value_kernel_m_read_readvariableop`
\savev2_adam_transformer_block_9_multi_head_self_attention_9_value_bias_m_read_readvariableop`
\savev2_adam_transformer_block_9_multi_head_self_attention_9_out_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_9_multi_head_self_attention_9_out_bias_m_read_readvariableop5
1savev2_adam_dense_36_kernel_m_read_readvariableop3
/savev2_adam_dense_36_bias_m_read_readvariableop5
1savev2_adam_dense_37_kernel_m_read_readvariableop3
/savev2_adam_dense_37_bias_m_read_readvariableopV
Rsavev2_adam_transformer_block_9_layer_normalization_32_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_9_layer_normalization_32_beta_m_read_readvariableopV
Rsavev2_adam_transformer_block_9_layer_normalization_33_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_9_layer_normalization_33_beta_m_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableopB
>savev2_adam_layer_normalization_31_gamma_v_read_readvariableopA
=savev2_adam_layer_normalization_31_beta_v_read_readvariableop5
1savev2_adam_dense_38_kernel_v_read_readvariableop3
/savev2_adam_dense_38_bias_v_read_readvariableopB
>savev2_adam_layer_normalization_34_gamma_v_read_readvariableopA
=savev2_adam_layer_normalization_34_beta_v_read_readvariableop5
1savev2_adam_dense_39_kernel_v_read_readvariableop3
/savev2_adam_dense_39_bias_v_read_readvariableopb
^savev2_adam_transformer_block_9_multi_head_self_attention_9_query_kernel_v_read_readvariableop`
\savev2_adam_transformer_block_9_multi_head_self_attention_9_query_bias_v_read_readvariableop`
\savev2_adam_transformer_block_9_multi_head_self_attention_9_key_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_9_multi_head_self_attention_9_key_bias_v_read_readvariableopb
^savev2_adam_transformer_block_9_multi_head_self_attention_9_value_kernel_v_read_readvariableop`
\savev2_adam_transformer_block_9_multi_head_self_attention_9_value_bias_v_read_readvariableop`
\savev2_adam_transformer_block_9_multi_head_self_attention_9_out_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_9_multi_head_self_attention_9_out_bias_v_read_readvariableop5
1savev2_adam_dense_36_kernel_v_read_readvariableop3
/savev2_adam_dense_36_bias_v_read_readvariableop5
1savev2_adam_dense_37_kernel_v_read_readvariableop3
/savev2_adam_dense_37_bias_v_read_readvariableopV
Rsavev2_adam_transformer_block_9_layer_normalization_32_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_9_layer_normalization_32_beta_v_read_readvariableopV
Rsavev2_adam_transformer_block_9_layer_normalization_33_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_9_layer_normalization_33_beta_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?,
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?+
value?+B?+XB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?
value?B?XB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?.
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop7savev2_layer_normalization_31_gamma_read_readvariableop6savev2_layer_normalization_31_beta_read_readvariableop*savev2_dense_38_kernel_read_readvariableop(savev2_dense_38_bias_read_readvariableop7savev2_layer_normalization_34_gamma_read_readvariableop6savev2_layer_normalization_34_beta_read_readvariableop*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopWsavev2_transformer_block_9_multi_head_self_attention_9_query_kernel_read_readvariableopUsavev2_transformer_block_9_multi_head_self_attention_9_query_bias_read_readvariableopUsavev2_transformer_block_9_multi_head_self_attention_9_key_kernel_read_readvariableopSsavev2_transformer_block_9_multi_head_self_attention_9_key_bias_read_readvariableopWsavev2_transformer_block_9_multi_head_self_attention_9_value_kernel_read_readvariableopUsavev2_transformer_block_9_multi_head_self_attention_9_value_bias_read_readvariableopUsavev2_transformer_block_9_multi_head_self_attention_9_out_kernel_read_readvariableopSsavev2_transformer_block_9_multi_head_self_attention_9_out_bias_read_readvariableop*savev2_dense_36_kernel_read_readvariableop(savev2_dense_36_bias_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableopKsavev2_transformer_block_9_layer_normalization_32_gamma_read_readvariableopJsavev2_transformer_block_9_layer_normalization_32_beta_read_readvariableopKsavev2_transformer_block_9_layer_normalization_33_gamma_read_readvariableopJsavev2_transformer_block_9_layer_normalization_33_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop>savev2_adam_layer_normalization_31_gamma_m_read_readvariableop=savev2_adam_layer_normalization_31_beta_m_read_readvariableop1savev2_adam_dense_38_kernel_m_read_readvariableop/savev2_adam_dense_38_bias_m_read_readvariableop>savev2_adam_layer_normalization_34_gamma_m_read_readvariableop=savev2_adam_layer_normalization_34_beta_m_read_readvariableop1savev2_adam_dense_39_kernel_m_read_readvariableop/savev2_adam_dense_39_bias_m_read_readvariableop^savev2_adam_transformer_block_9_multi_head_self_attention_9_query_kernel_m_read_readvariableop\savev2_adam_transformer_block_9_multi_head_self_attention_9_query_bias_m_read_readvariableop\savev2_adam_transformer_block_9_multi_head_self_attention_9_key_kernel_m_read_readvariableopZsavev2_adam_transformer_block_9_multi_head_self_attention_9_key_bias_m_read_readvariableop^savev2_adam_transformer_block_9_multi_head_self_attention_9_value_kernel_m_read_readvariableop\savev2_adam_transformer_block_9_multi_head_self_attention_9_value_bias_m_read_readvariableop\savev2_adam_transformer_block_9_multi_head_self_attention_9_out_kernel_m_read_readvariableopZsavev2_adam_transformer_block_9_multi_head_self_attention_9_out_bias_m_read_readvariableop1savev2_adam_dense_36_kernel_m_read_readvariableop/savev2_adam_dense_36_bias_m_read_readvariableop1savev2_adam_dense_37_kernel_m_read_readvariableop/savev2_adam_dense_37_bias_m_read_readvariableopRsavev2_adam_transformer_block_9_layer_normalization_32_gamma_m_read_readvariableopQsavev2_adam_transformer_block_9_layer_normalization_32_beta_m_read_readvariableopRsavev2_adam_transformer_block_9_layer_normalization_33_gamma_m_read_readvariableopQsavev2_adam_transformer_block_9_layer_normalization_33_beta_m_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableop>savev2_adam_layer_normalization_31_gamma_v_read_readvariableop=savev2_adam_layer_normalization_31_beta_v_read_readvariableop1savev2_adam_dense_38_kernel_v_read_readvariableop/savev2_adam_dense_38_bias_v_read_readvariableop>savev2_adam_layer_normalization_34_gamma_v_read_readvariableop=savev2_adam_layer_normalization_34_beta_v_read_readvariableop1savev2_adam_dense_39_kernel_v_read_readvariableop/savev2_adam_dense_39_bias_v_read_readvariableop^savev2_adam_transformer_block_9_multi_head_self_attention_9_query_kernel_v_read_readvariableop\savev2_adam_transformer_block_9_multi_head_self_attention_9_query_bias_v_read_readvariableop\savev2_adam_transformer_block_9_multi_head_self_attention_9_key_kernel_v_read_readvariableopZsavev2_adam_transformer_block_9_multi_head_self_attention_9_key_bias_v_read_readvariableop^savev2_adam_transformer_block_9_multi_head_self_attention_9_value_kernel_v_read_readvariableop\savev2_adam_transformer_block_9_multi_head_self_attention_9_value_bias_v_read_readvariableop\savev2_adam_transformer_block_9_multi_head_self_attention_9_out_kernel_v_read_readvariableopZsavev2_adam_transformer_block_9_multi_head_self_attention_9_out_bias_v_read_readvariableop1savev2_adam_dense_36_kernel_v_read_readvariableop/savev2_adam_dense_36_bias_v_read_readvariableop1savev2_adam_dense_37_kernel_v_read_readvariableop/savev2_adam_dense_37_bias_v_read_readvariableopRsavev2_adam_transformer_block_9_layer_normalization_32_gamma_v_read_readvariableopQsavev2_adam_transformer_block_9_layer_normalization_32_beta_v_read_readvariableopRsavev2_adam_transformer_block_9_layer_normalization_33_gamma_v_read_readvariableopQsavev2_adam_transformer_block_9_layer_normalization_33_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *f
dtypes\
Z2X	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :d@:@:@:@:	?:::::: : : : : :@::@::@:::::::::::: : : : :d@:@:@:@:	?::::::@::@::@::::::::::::d@:@:@:@:	?::::::@::@::@:::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:d@: %

_output_shapes
:@: &

_output_shapes
:@: '

_output_shapes
:@:%(!

_output_shapes
:	?: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::$. 

_output_shapes

:@: /

_output_shapes
::$0 

_output_shapes

:@: 1

_output_shapes
::$2 

_output_shapes

:@: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
::$6 

_output_shapes

:: 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
::$> 

_output_shapes

:d@: ?

_output_shapes
:@: @

_output_shapes
:@: A

_output_shapes
:@:%B!

_output_shapes
:	?: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::$F 

_output_shapes

:: G

_output_shapes
::$H 

_output_shapes

:@: I

_output_shapes
::$J 

_output_shapes

:@: K

_output_shapes
::$L 

_output_shapes

:@: M

_output_shapes
::$N 

_output_shapes

:: O

_output_shapes
::$P 

_output_shapes

:: Q

_output_shapes
::$R 

_output_shapes

:: S

_output_shapes
:: T

_output_shapes
:: U

_output_shapes
:: V

_output_shapes
:: W

_output_shapes
::X

_output_shapes
: 
??
?
C__inference_model_4_layer_call_and_return_conditional_losses_258431

inputs<
*dense_35_tensordot_readvariableop_resource:d@6
(dense_35_biasadd_readvariableop_resource:@B
4layer_normalization_31_mul_3_readvariableop_resource:@@
2layer_normalization_31_add_readvariableop_resource:@i
Wtransformer_block_9_multi_head_self_attention_9_query_tensordot_readvariableop_resource:@c
Utransformer_block_9_multi_head_self_attention_9_query_biasadd_readvariableop_resource:g
Utransformer_block_9_multi_head_self_attention_9_key_tensordot_readvariableop_resource:@a
Stransformer_block_9_multi_head_self_attention_9_key_biasadd_readvariableop_resource:i
Wtransformer_block_9_multi_head_self_attention_9_value_tensordot_readvariableop_resource:@c
Utransformer_block_9_multi_head_self_attention_9_value_biasadd_readvariableop_resource:g
Utransformer_block_9_multi_head_self_attention_9_out_tensordot_readvariableop_resource:a
Stransformer_block_9_multi_head_self_attention_9_out_biasadd_readvariableop_resource:^
Ptransformer_block_9_layer_normalization_32_batchnorm_mul_readvariableop_resource:Z
Ltransformer_block_9_layer_normalization_32_batchnorm_readvariableop_resource:]
Ktransformer_block_9_sequential_9_dense_36_tensordot_readvariableop_resource:W
Itransformer_block_9_sequential_9_dense_36_biasadd_readvariableop_resource:]
Ktransformer_block_9_sequential_9_dense_37_tensordot_readvariableop_resource:W
Itransformer_block_9_sequential_9_dense_37_biasadd_readvariableop_resource:^
Ptransformer_block_9_layer_normalization_33_batchnorm_mul_readvariableop_resource:Z
Ltransformer_block_9_layer_normalization_33_batchnorm_readvariableop_resource::
'dense_38_matmul_readvariableop_resource:	?6
(dense_38_biasadd_readvariableop_resource:B
4layer_normalization_34_mul_2_readvariableop_resource:@
2layer_normalization_34_add_readvariableop_resource:9
'dense_39_matmul_readvariableop_resource:6
(dense_39_biasadd_readvariableop_resource:
identity??dense_35/BiasAdd/ReadVariableOp?!dense_35/Tensordot/ReadVariableOp?1dense_35/kernel/Regularizer/Square/ReadVariableOp?dense_38/BiasAdd/ReadVariableOp?dense_38/MatMul/ReadVariableOp?1dense_38/kernel/Regularizer/Square/ReadVariableOp?dense_39/BiasAdd/ReadVariableOp?dense_39/MatMul/ReadVariableOp?1dense_39/kernel/Regularizer/Square/ReadVariableOp?)layer_normalization_31/add/ReadVariableOp?+layer_normalization_31/mul_3/ReadVariableOp?)layer_normalization_34/add/ReadVariableOp?+layer_normalization_34/mul_2/ReadVariableOp?Ctransformer_block_9/layer_normalization_32/batchnorm/ReadVariableOp?Gtransformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOp?Ctransformer_block_9/layer_normalization_33/batchnorm/ReadVariableOp?Gtransformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOp?Jtransformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOp?Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOp?Jtransformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOp?Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOp?Ltransformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOp?Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOp?Ltransformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOp?Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOp?@transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOp?Btransformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOp?@transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOp?Btransformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp?
!dense_35/Tensordot/ReadVariableOpReadVariableOp*dense_35_tensordot_readvariableop_resource*
_output_shapes

:d@*
dtype0a
dense_35/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_35/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_35/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_35/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_35/Tensordot/GatherV2GatherV2!dense_35/Tensordot/Shape:output:0 dense_35/Tensordot/free:output:0)dense_35/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_35/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_35/Tensordot/GatherV2_1GatherV2!dense_35/Tensordot/Shape:output:0 dense_35/Tensordot/axes:output:0+dense_35/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_35/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_35/Tensordot/ProdProd$dense_35/Tensordot/GatherV2:output:0!dense_35/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_35/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_35/Tensordot/Prod_1Prod&dense_35/Tensordot/GatherV2_1:output:0#dense_35/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_35/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_35/Tensordot/concatConcatV2 dense_35/Tensordot/free:output:0 dense_35/Tensordot/axes:output:0'dense_35/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_35/Tensordot/stackPack dense_35/Tensordot/Prod:output:0"dense_35/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_35/Tensordot/transpose	Transposeinputs"dense_35/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????pd?
dense_35/Tensordot/ReshapeReshape dense_35/Tensordot/transpose:y:0!dense_35/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_35/Tensordot/MatMulMatMul#dense_35/Tensordot/Reshape:output:0)dense_35/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
dense_35/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@b
 dense_35/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_35/Tensordot/concat_1ConcatV2$dense_35/Tensordot/GatherV2:output:0#dense_35/Tensordot/Const_2:output:0)dense_35/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_35/TensordotReshape#dense_35/Tensordot/MatMul:product:0$dense_35/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p@?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_35/BiasAddBiasAdddense_35/Tensordot:output:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@f
dense_35/ReluReludense_35/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p@]
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_27/dropout/MulMuldense_35/Relu:activations:0!dropout_27/dropout/Const:output:0*
T0*+
_output_shapes
:?????????p@c
dropout_27/dropout/ShapeShapedense_35/Relu:activations:0*
T0*
_output_shapes
:?
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????p@*
dtype0f
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????p@?
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????p@?
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????p@h
layer_normalization_31/ShapeShapedropout_27/dropout/Mul_1:z:0*
T0*
_output_shapes
:t
*layer_normalization_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_31/strided_sliceStridedSlice%layer_normalization_31/Shape:output:03layer_normalization_31/strided_slice/stack:output:05layer_normalization_31/strided_slice/stack_1:output:05layer_normalization_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_31/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_31/mulMul%layer_normalization_31/mul/x:output:0-layer_normalization_31/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_31/strided_slice_1StridedSlice%layer_normalization_31/Shape:output:05layer_normalization_31/strided_slice_1/stack:output:07layer_normalization_31/strided_slice_1/stack_1:output:07layer_normalization_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_31/mul_1Mullayer_normalization_31/mul:z:0/layer_normalization_31/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_31/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_31/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_31/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_31/strided_slice_2StridedSlice%layer_normalization_31/Shape:output:05layer_normalization_31/strided_slice_2/stack:output:07layer_normalization_31/strided_slice_2/stack_1:output:07layer_normalization_31/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_31/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_31/mul_2Mul'layer_normalization_31/mul_2/x:output:0/layer_normalization_31/strided_slice_2:output:0*
T0*
_output_shapes
: h
&layer_normalization_31/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_31/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_31/Reshape/shapePack/layer_normalization_31/Reshape/shape/0:output:0 layer_normalization_31/mul_1:z:0 layer_normalization_31/mul_2:z:0/layer_normalization_31/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_31/ReshapeReshapedropout_27/dropout/Mul_1:z:0-layer_normalization_31/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@z
"layer_normalization_31/ones/packedPack layer_normalization_31/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_31/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_31/onesFill+layer_normalization_31/ones/packed:output:0*layer_normalization_31/ones/Const:output:0*
T0*#
_output_shapes
:?????????{
#layer_normalization_31/zeros/packedPack layer_normalization_31/mul_1:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_31/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_31/zerosFill,layer_normalization_31/zeros/packed:output:0+layer_normalization_31/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_31/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_31/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_31/FusedBatchNormV3FusedBatchNormV3'layer_normalization_31/Reshape:output:0$layer_normalization_31/ones:output:0%layer_normalization_31/zeros:output:0%layer_normalization_31/Const:output:0'layer_normalization_31/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
 layer_normalization_31/Reshape_1Reshape+layer_normalization_31/FusedBatchNormV3:y:0%layer_normalization_31/Shape:output:0*
T0*+
_output_shapes
:?????????p@?
+layer_normalization_31/mul_3/ReadVariableOpReadVariableOp4layer_normalization_31_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization_31/mul_3Mul)layer_normalization_31/Reshape_1:output:03layer_normalization_31/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@?
)layer_normalization_31/add/ReadVariableOpReadVariableOp2layer_normalization_31_add_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization_31/addAddV2 layer_normalization_31/mul_3:z:01layer_normalization_31/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@?
5transformer_block_9/multi_head_self_attention_9/ShapeShapelayer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Ctransformer_block_9/multi_head_self_attention_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Etransformer_block_9/multi_head_self_attention_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Etransformer_block_9/multi_head_self_attention_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=transformer_block_9/multi_head_self_attention_9/strided_sliceStridedSlice>transformer_block_9/multi_head_self_attention_9/Shape:output:0Ltransformer_block_9/multi_head_self_attention_9/strided_slice/stack:output:0Ntransformer_block_9/multi_head_self_attention_9/strided_slice/stack_1:output:0Ntransformer_block_9/multi_head_self_attention_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOpReadVariableOpWtransformer_block_9_multi_head_self_attention_9_query_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0?
Dtransformer_block_9/multi_head_self_attention_9/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Dtransformer_block_9/multi_head_self_attention_9/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Etransformer_block_9/multi_head_self_attention_9/query/Tensordot/ShapeShapelayer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2GatherV2Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/Shape:output:0Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/free:output:0Vtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Otransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Jtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1GatherV2Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/Shape:output:0Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/axes:output:0Xtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Etransformer_block_9/multi_head_self_attention_9/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_9/multi_head_self_attention_9/query/Tensordot/ProdProdQtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2:output:0Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Gtransformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Ftransformer_block_9/multi_head_self_attention_9/query/Tensordot/Prod_1ProdStransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1:output:0Ptransformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Ktransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/query/Tensordot/concatConcatV2Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/free:output:0Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/axes:output:0Ttransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Etransformer_block_9/multi_head_self_attention_9/query/Tensordot/stackPackMtransformer_block_9/multi_head_self_attention_9/query/Tensordot/Prod:output:0Otransformer_block_9/multi_head_self_attention_9/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Itransformer_block_9/multi_head_self_attention_9/query/Tensordot/transpose	Transposelayer_normalization_31/add:z:0Otransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
Gtransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReshapeReshapeMtransformer_block_9/multi_head_self_attention_9/query/Tensordot/transpose:y:0Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Ftransformer_block_9/multi_head_self_attention_9/query/Tensordot/MatMulMatMulPtransformer_block_9/multi_head_self_attention_9/query/Tensordot/Reshape:output:0Vtransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Gtransformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1ConcatV2Qtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2:output:0Ptransformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_2:output:0Vtransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
?transformer_block_9/multi_head_self_attention_9/query/TensordotReshapePtransformer_block_9/multi_head_self_attention_9/query/Tensordot/MatMul:product:0Qtransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Ltransformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOpReadVariableOpUtransformer_block_9_multi_head_self_attention_9_query_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
=transformer_block_9/multi_head_self_attention_9/query/BiasAddBiasAddHtransformer_block_9/multi_head_self_attention_9/query/Tensordot:output:0Ttransformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_9_multi_head_self_attention_9_key_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0?
Btransformer_block_9/multi_head_self_attention_9/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_9/multi_head_self_attention_9/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_9/multi_head_self_attention_9/key/Tensordot/ShapeShapelayer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2GatherV2Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/Shape:output:0Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/free:output:0Ttransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1GatherV2Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/Shape:output:0Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/axes:output:0Vtransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_9/multi_head_self_attention_9/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_9/multi_head_self_attention_9/key/Tensordot/ProdProdOtransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2:output:0Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_9/multi_head_self_attention_9/key/Tensordot/Prod_1ProdQtransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1:output:0Ntransformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_9/multi_head_self_attention_9/key/Tensordot/concatConcatV2Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/free:output:0Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/axes:output:0Rtransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_9/multi_head_self_attention_9/key/Tensordot/stackPackKtransformer_block_9/multi_head_self_attention_9/key/Tensordot/Prod:output:0Mtransformer_block_9/multi_head_self_attention_9/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_9/multi_head_self_attention_9/key/Tensordot/transpose	Transposelayer_normalization_31/add:z:0Mtransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
Etransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReshapeReshapeKtransformer_block_9/multi_head_self_attention_9/key/Tensordot/transpose:y:0Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_9/multi_head_self_attention_9/key/Tensordot/MatMulMatMulNtransformer_block_9/multi_head_self_attention_9/key/Tensordot/Reshape:output:0Ttransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1ConcatV2Otransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2:output:0Ntransformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_2:output:0Ttransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_9/multi_head_self_attention_9/key/TensordotReshapeNtransformer_block_9/multi_head_self_attention_9/key/Tensordot/MatMul:product:0Otransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Jtransformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_9_multi_head_self_attention_9_key_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_9/multi_head_self_attention_9/key/BiasAddBiasAddFtransformer_block_9/multi_head_self_attention_9/key/Tensordot:output:0Rtransformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOpReadVariableOpWtransformer_block_9_multi_head_self_attention_9_value_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0?
Dtransformer_block_9/multi_head_self_attention_9/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Dtransformer_block_9/multi_head_self_attention_9/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Etransformer_block_9/multi_head_self_attention_9/value/Tensordot/ShapeShapelayer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2GatherV2Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/Shape:output:0Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/free:output:0Vtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Otransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Jtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1GatherV2Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/Shape:output:0Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/axes:output:0Xtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Etransformer_block_9/multi_head_self_attention_9/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_9/multi_head_self_attention_9/value/Tensordot/ProdProdQtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2:output:0Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Gtransformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Ftransformer_block_9/multi_head_self_attention_9/value/Tensordot/Prod_1ProdStransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1:output:0Ptransformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Ktransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/value/Tensordot/concatConcatV2Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/free:output:0Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/axes:output:0Ttransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Etransformer_block_9/multi_head_self_attention_9/value/Tensordot/stackPackMtransformer_block_9/multi_head_self_attention_9/value/Tensordot/Prod:output:0Otransformer_block_9/multi_head_self_attention_9/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Itransformer_block_9/multi_head_self_attention_9/value/Tensordot/transpose	Transposelayer_normalization_31/add:z:0Otransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
Gtransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReshapeReshapeMtransformer_block_9/multi_head_self_attention_9/value/Tensordot/transpose:y:0Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Ftransformer_block_9/multi_head_self_attention_9/value/Tensordot/MatMulMatMulPtransformer_block_9/multi_head_self_attention_9/value/Tensordot/Reshape:output:0Vtransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Gtransformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1ConcatV2Qtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2:output:0Ptransformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_2:output:0Vtransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
?transformer_block_9/multi_head_self_attention_9/value/TensordotReshapePtransformer_block_9/multi_head_self_attention_9/value/Tensordot/MatMul:product:0Qtransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Ltransformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOpReadVariableOpUtransformer_block_9_multi_head_self_attention_9_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
=transformer_block_9/multi_head_self_attention_9/value/BiasAddBiasAddHtransformer_block_9/multi_head_self_attention_9/value/Tensordot:output:0Ttransformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
?transformer_block_9/multi_head_self_attention_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
?transformer_block_9/multi_head_self_attention_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
?transformer_block_9/multi_head_self_attention_9/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
=transformer_block_9/multi_head_self_attention_9/Reshape/shapePackFtransformer_block_9/multi_head_self_attention_9/strided_slice:output:0Htransformer_block_9/multi_head_self_attention_9/Reshape/shape/1:output:0Htransformer_block_9/multi_head_self_attention_9/Reshape/shape/2:output:0Htransformer_block_9/multi_head_self_attention_9/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
7transformer_block_9/multi_head_self_attention_9/ReshapeReshapeFtransformer_block_9/multi_head_self_attention_9/query/BiasAdd:output:0Ftransformer_block_9/multi_head_self_attention_9/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????p?
>transformer_block_9/multi_head_self_attention_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
9transformer_block_9/multi_head_self_attention_9/transpose	Transpose@transformer_block_9/multi_head_self_attention_9/Reshape:output:0Gtransformer_block_9/multi_head_self_attention_9/transpose/perm:output:0*
T0*/
_output_shapes
:?????????p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Atransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
?transformer_block_9/multi_head_self_attention_9/Reshape_1/shapePackFtransformer_block_9/multi_head_self_attention_9/strided_slice:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/1:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/2:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_9/multi_head_self_attention_9/Reshape_1ReshapeDtransformer_block_9/multi_head_self_attention_9/key/BiasAdd:output:0Htransformer_block_9/multi_head_self_attention_9/Reshape_1/shape:output:0*
T0*/
_output_shapes
:?????????p?
@transformer_block_9/multi_head_self_attention_9/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
;transformer_block_9/multi_head_self_attention_9/transpose_1	TransposeBtransformer_block_9/multi_head_self_attention_9/Reshape_1:output:0Itransformer_block_9/multi_head_self_attention_9/transpose_1/perm:output:0*
T0*/
_output_shapes
:?????????p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Atransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
?transformer_block_9/multi_head_self_attention_9/Reshape_2/shapePackFtransformer_block_9/multi_head_self_attention_9/strided_slice:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/1:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/2:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_9/multi_head_self_attention_9/Reshape_2ReshapeFtransformer_block_9/multi_head_self_attention_9/value/BiasAdd:output:0Htransformer_block_9/multi_head_self_attention_9/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????p?
@transformer_block_9/multi_head_self_attention_9/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
;transformer_block_9/multi_head_self_attention_9/transpose_2	TransposeBtransformer_block_9/multi_head_self_attention_9/Reshape_2:output:0Itransformer_block_9/multi_head_self_attention_9/transpose_2/perm:output:0*
T0*/
_output_shapes
:?????????p?
6transformer_block_9/multi_head_self_attention_9/MatMulBatchMatMulV2=transformer_block_9/multi_head_self_attention_9/transpose:y:0?transformer_block_9/multi_head_self_attention_9/transpose_1:y:0*
T0*/
_output_shapes
:?????????pp*
adj_y(?
7transformer_block_9/multi_head_self_attention_9/Shape_1Shape?transformer_block_9/multi_head_self_attention_9/transpose_1:y:0*
T0*
_output_shapes
:?
Etransformer_block_9/multi_head_self_attention_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Gtransformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Gtransformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?transformer_block_9/multi_head_self_attention_9/strided_slice_1StridedSlice@transformer_block_9/multi_head_self_attention_9/Shape_1:output:0Ntransformer_block_9/multi_head_self_attention_9/strided_slice_1/stack:output:0Ptransformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_1:output:0Ptransformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
4transformer_block_9/multi_head_self_attention_9/CastCastHtransformer_block_9/multi_head_self_attention_9/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
4transformer_block_9/multi_head_self_attention_9/SqrtSqrt8transformer_block_9/multi_head_self_attention_9/Cast:y:0*
T0*
_output_shapes
: ?
7transformer_block_9/multi_head_self_attention_9/truedivRealDiv?transformer_block_9/multi_head_self_attention_9/MatMul:output:08transformer_block_9/multi_head_self_attention_9/Sqrt:y:0*
T0*/
_output_shapes
:?????????pp?
7transformer_block_9/multi_head_self_attention_9/SoftmaxSoftmax;transformer_block_9/multi_head_self_attention_9/truediv:z:0*
T0*/
_output_shapes
:?????????pp?
8transformer_block_9/multi_head_self_attention_9/MatMul_1BatchMatMulV2Atransformer_block_9/multi_head_self_attention_9/Softmax:softmax:0?transformer_block_9/multi_head_self_attention_9/transpose_2:y:0*
T0*/
_output_shapes
:?????????p?
@transformer_block_9/multi_head_self_attention_9/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
;transformer_block_9/multi_head_self_attention_9/transpose_3	TransposeAtransformer_block_9/multi_head_self_attention_9/MatMul_1:output:0Itransformer_block_9/multi_head_self_attention_9/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
?transformer_block_9/multi_head_self_attention_9/Reshape_3/shapePackFtransformer_block_9/multi_head_self_attention_9/strided_slice:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_3/shape/1:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_9/multi_head_self_attention_9/Reshape_3Reshape?transformer_block_9/multi_head_self_attention_9/transpose_3:y:0Htransformer_block_9/multi_head_self_attention_9/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????p?
Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_9_multi_head_self_attention_9_out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Btransformer_block_9/multi_head_self_attention_9/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_9/multi_head_self_attention_9/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_9/multi_head_self_attention_9/out/Tensordot/ShapeShapeBtransformer_block_9/multi_head_self_attention_9/Reshape_3:output:0*
T0*
_output_shapes
:?
Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2GatherV2Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/Shape:output:0Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/free:output:0Ttransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1GatherV2Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/Shape:output:0Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/axes:output:0Vtransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_9/multi_head_self_attention_9/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_9/multi_head_self_attention_9/out/Tensordot/ProdProdOtransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2:output:0Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_9/multi_head_self_attention_9/out/Tensordot/Prod_1ProdQtransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1:output:0Ntransformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_9/multi_head_self_attention_9/out/Tensordot/concatConcatV2Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/free:output:0Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/axes:output:0Rtransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_9/multi_head_self_attention_9/out/Tensordot/stackPackKtransformer_block_9/multi_head_self_attention_9/out/Tensordot/Prod:output:0Mtransformer_block_9/multi_head_self_attention_9/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_9/multi_head_self_attention_9/out/Tensordot/transpose	TransposeBtransformer_block_9/multi_head_self_attention_9/Reshape_3:output:0Mtransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
Etransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReshapeReshapeKtransformer_block_9/multi_head_self_attention_9/out/Tensordot/transpose:y:0Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_9/multi_head_self_attention_9/out/Tensordot/MatMulMatMulNtransformer_block_9/multi_head_self_attention_9/out/Tensordot/Reshape:output:0Ttransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1ConcatV2Otransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2:output:0Ntransformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_2:output:0Ttransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_9/multi_head_self_attention_9/out/TensordotReshapeNtransformer_block_9/multi_head_self_attention_9/out/Tensordot/MatMul:product:0Otransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Jtransformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_9_multi_head_self_attention_9_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_9/multi_head_self_attention_9/out/BiasAddBiasAddFtransformer_block_9/multi_head_self_attention_9/out/Tensordot:output:0Rtransformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pq
,transformer_block_9/dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
*transformer_block_9/dropout_28/dropout/MulMulDtransformer_block_9/multi_head_self_attention_9/out/BiasAdd:output:05transformer_block_9/dropout_28/dropout/Const:output:0*
T0*+
_output_shapes
:?????????p?
,transformer_block_9/dropout_28/dropout/ShapeShapeDtransformer_block_9/multi_head_self_attention_9/out/BiasAdd:output:0*
T0*
_output_shapes
:?
Ctransformer_block_9/dropout_28/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_9/dropout_28/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????p*
dtype0z
5transformer_block_9/dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
3transformer_block_9/dropout_28/dropout/GreaterEqualGreaterEqualLtransformer_block_9/dropout_28/dropout/random_uniform/RandomUniform:output:0>transformer_block_9/dropout_28/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????p?
+transformer_block_9/dropout_28/dropout/CastCast7transformer_block_9/dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????p?
,transformer_block_9/dropout_28/dropout/Mul_1Mul.transformer_block_9/dropout_28/dropout/Mul:z:0/transformer_block_9/dropout_28/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????p?
Itransformer_block_9/layer_normalization_32/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_block_9/layer_normalization_32/moments/meanMean0transformer_block_9/dropout_28/dropout/Mul_1:z:0Rtransformer_block_9/layer_normalization_32/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
?transformer_block_9/layer_normalization_32/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_32/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
Dtransformer_block_9/layer_normalization_32/moments/SquaredDifferenceSquaredDifference0transformer_block_9/dropout_28/dropout/Mul_1:z:0Htransformer_block_9/layer_normalization_32/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
Mtransformer_block_9/layer_normalization_32/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_9/layer_normalization_32/moments/varianceMeanHtransformer_block_9/layer_normalization_32/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_32/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(
:transformer_block_9/layer_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
8transformer_block_9/layer_normalization_32/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_32/moments/variance:output:0Ctransformer_block_9/layer_normalization_32/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_32/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_32/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
Gtransformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_block_9/layer_normalization_32/batchnorm/mulMul>transformer_block_9/layer_normalization_32/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_32/batchnorm/mul_1Mul0transformer_block_9/dropout_28/dropout/Mul_1:z:0<transformer_block_9/layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_32/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_32/moments/mean:output:0<transformer_block_9/layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
Ctransformer_block_9/layer_normalization_32/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_block_9/layer_normalization_32/batchnorm/subSubKtransformer_block_9/layer_normalization_32/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_32/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_32/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_32/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_32/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p?
Btransformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_36_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
8transformer_block_9/sequential_9/dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_block_9/sequential_9/dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
9transformer_block_9/sequential_9/dense_36/Tensordot/ShapeShape>transformer_block_9/layer_normalization_32/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
Atransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_36/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_36/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_36/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_36/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
9transformer_block_9/sequential_9/dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
8transformer_block_9/sequential_9/dense_36/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
;transformer_block_9/sequential_9/dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_9/sequential_9/dense_36/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
?transformer_block_9/sequential_9/dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:transformer_block_9/sequential_9/dense_36/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_36/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_36/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_9/sequential_9/dense_36/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_36/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_9/sequential_9/dense_36/Tensordot/transpose	Transpose>transformer_block_9/layer_normalization_32/batchnorm/add_1:z:0Ctransformer_block_9/sequential_9/dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
;transformer_block_9/sequential_9/dense_36/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_36/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
:transformer_block_9/sequential_9/dense_36/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_36/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;transformer_block_9/sequential_9/dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_9/sequential_9/dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_9/sequential_9/dense_36/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_36/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_9/sequential_9/dense_36/TensordotReshapeDtransformer_block_9/sequential_9/dense_36/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
@transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1transformer_block_9/sequential_9/dense_36/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_36/Tensordot:output:0Htransformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
.transformer_block_9/sequential_9/dense_36/ReluRelu:transformer_block_9/sequential_9/dense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
Btransformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_37_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
8transformer_block_9/sequential_9/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_block_9/sequential_9/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
9transformer_block_9/sequential_9/dense_37/Tensordot/ShapeShape<transformer_block_9/sequential_9/dense_36/Relu:activations:0*
T0*
_output_shapes
:?
Atransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_37/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_37/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_37/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_37/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
9transformer_block_9/sequential_9/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
8transformer_block_9/sequential_9/dense_37/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
;transformer_block_9/sequential_9/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_9/sequential_9/dense_37/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
?transformer_block_9/sequential_9/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:transformer_block_9/sequential_9/dense_37/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_37/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_37/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_9/sequential_9/dense_37/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_37/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_9/sequential_9/dense_37/Tensordot/transpose	Transpose<transformer_block_9/sequential_9/dense_36/Relu:activations:0Ctransformer_block_9/sequential_9/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
;transformer_block_9/sequential_9/dense_37/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_37/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
:transformer_block_9/sequential_9/dense_37/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_37/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;transformer_block_9/sequential_9/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_9/sequential_9/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_9/sequential_9/dense_37/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_37/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_9/sequential_9/dense_37/TensordotReshapeDtransformer_block_9/sequential_9/dense_37/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
@transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1transformer_block_9/sequential_9/dense_37/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_37/Tensordot:output:0Htransformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pq
,transformer_block_9/dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
*transformer_block_9/dropout_29/dropout/MulMul:transformer_block_9/sequential_9/dense_37/BiasAdd:output:05transformer_block_9/dropout_29/dropout/Const:output:0*
T0*+
_output_shapes
:?????????p?
,transformer_block_9/dropout_29/dropout/ShapeShape:transformer_block_9/sequential_9/dense_37/BiasAdd:output:0*
T0*
_output_shapes
:?
Ctransformer_block_9/dropout_29/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_9/dropout_29/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????p*
dtype0z
5transformer_block_9/dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
3transformer_block_9/dropout_29/dropout/GreaterEqualGreaterEqualLtransformer_block_9/dropout_29/dropout/random_uniform/RandomUniform:output:0>transformer_block_9/dropout_29/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????p?
+transformer_block_9/dropout_29/dropout/CastCast7transformer_block_9/dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????p?
,transformer_block_9/dropout_29/dropout/Mul_1Mul.transformer_block_9/dropout_29/dropout/Mul:z:0/transformer_block_9/dropout_29/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????p?
transformer_block_9/addAddV2>transformer_block_9/layer_normalization_32/batchnorm/add_1:z:00transformer_block_9/dropout_29/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????p?
Itransformer_block_9/layer_normalization_33/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_block_9/layer_normalization_33/moments/meanMeantransformer_block_9/add:z:0Rtransformer_block_9/layer_normalization_33/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
?transformer_block_9/layer_normalization_33/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_33/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
Dtransformer_block_9/layer_normalization_33/moments/SquaredDifferenceSquaredDifferencetransformer_block_9/add:z:0Htransformer_block_9/layer_normalization_33/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
Mtransformer_block_9/layer_normalization_33/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_9/layer_normalization_33/moments/varianceMeanHtransformer_block_9/layer_normalization_33/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_33/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(
:transformer_block_9/layer_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
8transformer_block_9/layer_normalization_33/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_33/moments/variance:output:0Ctransformer_block_9/layer_normalization_33/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_33/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_33/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
Gtransformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_block_9/layer_normalization_33/batchnorm/mulMul>transformer_block_9/layer_normalization_33/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_33/batchnorm/mul_1Multransformer_block_9/add:z:0<transformer_block_9/layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_33/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_33/moments/mean:output:0<transformer_block_9/layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
Ctransformer_block_9/layer_normalization_33/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_block_9/layer_normalization_33/batchnorm/subSubKtransformer_block_9/layer_normalization_33/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_33/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_33/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_33/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_33/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_4/ReshapeReshape>transformer_block_9/layer_normalization_33/batchnorm/add_1:z:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????]
dropout_30/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU???
dropout_30/dropout/MulMulflatten_4/Reshape:output:0!dropout_30/dropout/Const:output:0*
T0*(
_output_shapes
:??????????b
dropout_30/dropout/ShapeShapeflatten_4/Reshape:output:0*
T0*
_output_shapes
:?
/dropout_30/dropout/random_uniform/RandomUniformRandomUniform!dropout_30/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0f
!dropout_30/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_30/dropout/GreaterEqualGreaterEqual8dropout_30/dropout/random_uniform/RandomUniform:output:0*dropout_30/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_30/dropout/CastCast#dropout_30/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_30/dropout/Mul_1Muldropout_30/dropout/Mul:z:0dropout_30/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_38/MatMulMatMuldropout_30/dropout/Mul_1:z:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
layer_normalization_34/ShapeShapedense_38/Relu:activations:0*
T0*
_output_shapes
:t
*layer_normalization_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_34/strided_sliceStridedSlice%layer_normalization_34/Shape:output:03layer_normalization_34/strided_slice/stack:output:05layer_normalization_34/strided_slice/stack_1:output:05layer_normalization_34/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_34/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_34/mulMul%layer_normalization_34/mul/x:output:0-layer_normalization_34/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_34/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_34/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_34/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_34/strided_slice_1StridedSlice%layer_normalization_34/Shape:output:05layer_normalization_34/strided_slice_1/stack:output:07layer_normalization_34/strided_slice_1/stack_1:output:07layer_normalization_34/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_34/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_34/mul_1Mul'layer_normalization_34/mul_1/x:output:0/layer_normalization_34/strided_slice_1:output:0*
T0*
_output_shapes
: h
&layer_normalization_34/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_34/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_34/Reshape/shapePack/layer_normalization_34/Reshape/shape/0:output:0layer_normalization_34/mul:z:0 layer_normalization_34/mul_1:z:0/layer_normalization_34/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_34/ReshapeReshapedense_38/Relu:activations:0-layer_normalization_34/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????x
"layer_normalization_34/ones/packedPacklayer_normalization_34/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_34/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_34/onesFill+layer_normalization_34/ones/packed:output:0*layer_normalization_34/ones/Const:output:0*
T0*#
_output_shapes
:?????????y
#layer_normalization_34/zeros/packedPacklayer_normalization_34/mul:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_34/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_34/zerosFill,layer_normalization_34/zeros/packed:output:0+layer_normalization_34/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_34/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_34/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_34/FusedBatchNormV3FusedBatchNormV3'layer_normalization_34/Reshape:output:0$layer_normalization_34/ones:output:0%layer_normalization_34/zeros:output:0%layer_normalization_34/Const:output:0'layer_normalization_34/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
 layer_normalization_34/Reshape_1Reshape+layer_normalization_34/FusedBatchNormV3:y:0%layer_normalization_34/Shape:output:0*
T0*'
_output_shapes
:??????????
+layer_normalization_34/mul_2/ReadVariableOpReadVariableOp4layer_normalization_34_mul_2_readvariableop_resource*
_output_shapes
:*
dtype0?
layer_normalization_34/mul_2Mul)layer_normalization_34/Reshape_1:output:03layer_normalization_34/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)layer_normalization_34/add/ReadVariableOpReadVariableOp2layer_normalization_34_add_readvariableop_resource*
_output_shapes
:*
dtype0?
layer_normalization_34/addAddV2 layer_normalization_34/mul_2:z:01layer_normalization_34/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_39/MatMulMatMullayer_normalization_34/add:z:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_39/ReluReludense_39/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_35_tensordot_readvariableop_resource*
_output_shapes

:d@*
dtype0?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d@r
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_35/kernel/Regularizer/SumSum&dense_35/kernel/Regularizer/Square:y:0*dense_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?r
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentitydense_39/Relu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_35/BiasAdd/ReadVariableOp"^dense_35/Tensordot/ReadVariableOp2^dense_35/kernel/Regularizer/Square/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp2^dense_39/kernel/Regularizer/Square/ReadVariableOp*^layer_normalization_31/add/ReadVariableOp,^layer_normalization_31/mul_3/ReadVariableOp*^layer_normalization_34/add/ReadVariableOp,^layer_normalization_34/mul_2/ReadVariableOpD^transformer_block_9/layer_normalization_32/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOpD^transformer_block_9/layer_normalization_33/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOpK^transformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOpM^transformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOpK^transformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOpM^transformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOpM^transformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOpO^transformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOpM^transformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOpO^transformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOpA^transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOpA^transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2F
!dense_35/Tensordot/ReadVariableOp!dense_35/Tensordot/ReadVariableOp2f
1dense_35/kernel/Regularizer/Square/ReadVariableOp1dense_35/kernel/Regularizer/Square/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2f
1dense_39/kernel/Regularizer/Square/ReadVariableOp1dense_39/kernel/Regularizer/Square/ReadVariableOp2V
)layer_normalization_31/add/ReadVariableOp)layer_normalization_31/add/ReadVariableOp2Z
+layer_normalization_31/mul_3/ReadVariableOp+layer_normalization_31/mul_3/ReadVariableOp2V
)layer_normalization_34/add/ReadVariableOp)layer_normalization_34/add/ReadVariableOp2Z
+layer_normalization_34/mul_2/ReadVariableOp+layer_normalization_34/mul_2/ReadVariableOp2?
Ctransformer_block_9/layer_normalization_32/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_32/batchnorm/ReadVariableOp2?
Gtransformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOp2?
Ctransformer_block_9/layer_normalization_33/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_33/batchnorm/ReadVariableOp2?
Gtransformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOp2?
Jtransformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOpJtransformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOp2?
Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOpLtransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOp2?
Jtransformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOpJtransformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOp2?
Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOpLtransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOp2?
Ltransformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOpLtransformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOp2?
Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOpNtransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOp2?
Ltransformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOpLtransformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOp2?
Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOpNtransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOp2?
@transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOp2?
Btransformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOp2?
@transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOp2?
Btransformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????pd
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_255966
dense_36_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_255955s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????p: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:?????????p
(
_user_specified_namedense_36_input
?
?
(__inference_model_4_layer_call_fn_257629

inputs
unknown:d@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:	?

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_257213o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????pd
 
_user_specified_nameinputs
?
G
+__inference_dropout_30_layer_call_fn_259215

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_256467a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_dense_38_layer_call_and_return_conditional_losses_259269

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_38/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?r
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_layer_normalization_34_layer_call_and_return_conditional_losses_256534

inputs+
mul_2_readvariableop_resource:)
add_readvariableop_resource:
identity??add/ReadVariableOp?mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
C__inference_model_4_layer_call_and_return_conditional_losses_256582

inputs!
dense_35_256112:d@
dense_35_256114:@+
layer_normalization_31_256172:@+
layer_normalization_31_256174:@,
transformer_block_9_256421:@(
transformer_block_9_256423:,
transformer_block_9_256425:@(
transformer_block_9_256427:,
transformer_block_9_256429:@(
transformer_block_9_256431:,
transformer_block_9_256433:(
transformer_block_9_256435:(
transformer_block_9_256437:(
transformer_block_9_256439:,
transformer_block_9_256441:(
transformer_block_9_256443:,
transformer_block_9_256445:(
transformer_block_9_256447:(
transformer_block_9_256449:(
transformer_block_9_256451:"
dense_38_256487:	?
dense_38_256489:+
layer_normalization_34_256535:+
layer_normalization_34_256537:!
dense_39_256558:
dense_39_256560:
identity?? dense_35/StatefulPartitionedCall?1dense_35/kernel/Regularizer/Square/ReadVariableOp? dense_38/StatefulPartitionedCall?1dense_38/kernel/Regularizer/Square/ReadVariableOp? dense_39/StatefulPartitionedCall?1dense_39/kernel/Regularizer/Square/ReadVariableOp?.layer_normalization_31/StatefulPartitionedCall?.layer_normalization_34/StatefulPartitionedCall?+transformer_block_9/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCallinputsdense_35_256112dense_35_256114*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_256111?
dropout_27/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_256122?
.layer_normalization_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0layer_normalization_31_256172layer_normalization_31_256174*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_31_layer_call_and_return_conditional_losses_256171?
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_31/StatefulPartitionedCall:output:0transformer_block_9_256421transformer_block_9_256423transformer_block_9_256425transformer_block_9_256427transformer_block_9_256429transformer_block_9_256431transformer_block_9_256433transformer_block_9_256435transformer_block_9_256437transformer_block_9_256439transformer_block_9_256441transformer_block_9_256443transformer_block_9_256445transformer_block_9_256447transformer_block_9_256449transformer_block_9_256451*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_256420?
flatten_4/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_256460?
dropout_30/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_256467?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0dense_38_256487dense_38_256489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_256486?
.layer_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0layer_normalization_34_256535layer_normalization_34_256537*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_34_layer_call_and_return_conditional_losses_256534?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_34/StatefulPartitionedCall:output:0dense_39_256558dense_39_256560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_256557?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_256112*
_output_shapes

:d@*
dtype0?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d@r
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_35/kernel/Regularizer/SumSum&dense_35/kernel/Regularizer/Square:y:0*dense_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38_256487*
_output_shapes
:	?*
dtype0?
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?r
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_39_256558*
_output_shapes

:*
dtype0?
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_35/StatefulPartitionedCall2^dense_35/kernel/Regularizer/Square/ReadVariableOp!^dense_38/StatefulPartitionedCall2^dense_38/kernel/Regularizer/Square/ReadVariableOp!^dense_39/StatefulPartitionedCall2^dense_39/kernel/Regularizer/Square/ReadVariableOp/^layer_normalization_31/StatefulPartitionedCall/^layer_normalization_34/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2f
1dense_35/kernel/Regularizer/Square/ReadVariableOp1dense_35/kernel/Regularizer/Square/ReadVariableOp2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2f
1dense_39/kernel/Regularizer/Square/ReadVariableOp1dense_39/kernel/Regularizer/Square/ReadVariableOp2`
.layer_normalization_31/StatefulPartitionedCall.layer_normalization_31/StatefulPartitionedCall2`
.layer_normalization_34/StatefulPartitionedCall.layer_normalization_34/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:S O
+
_output_shapes
:?????????pd
 
_user_specified_nameinputs
?
?
7__inference_layer_normalization_31_layer_call_fn_258578

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_31_layer_call_and_return_conditional_losses_256171s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????p@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_256015

inputs!
dense_36_256004:
dense_36_256006:!
dense_37_256009:
dense_37_256011:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_256004dense_36_256006*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_255912?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_256009dense_37_256011*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_255948|
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????p: : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
??
?
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_256420

inputsU
Cmulti_head_self_attention_9_query_tensordot_readvariableop_resource:@O
Amulti_head_self_attention_9_query_biasadd_readvariableop_resource:S
Amulti_head_self_attention_9_key_tensordot_readvariableop_resource:@M
?multi_head_self_attention_9_key_biasadd_readvariableop_resource:U
Cmulti_head_self_attention_9_value_tensordot_readvariableop_resource:@O
Amulti_head_self_attention_9_value_biasadd_readvariableop_resource:S
Amulti_head_self_attention_9_out_tensordot_readvariableop_resource:M
?multi_head_self_attention_9_out_biasadd_readvariableop_resource:J
<layer_normalization_32_batchnorm_mul_readvariableop_resource:F
8layer_normalization_32_batchnorm_readvariableop_resource:I
7sequential_9_dense_36_tensordot_readvariableop_resource:C
5sequential_9_dense_36_biasadd_readvariableop_resource:I
7sequential_9_dense_37_tensordot_readvariableop_resource:C
5sequential_9_dense_37_biasadd_readvariableop_resource:J
<layer_normalization_33_batchnorm_mul_readvariableop_resource:F
8layer_normalization_33_batchnorm_readvariableop_resource:
identity??/layer_normalization_32/batchnorm/ReadVariableOp?3layer_normalization_32/batchnorm/mul/ReadVariableOp?/layer_normalization_33/batchnorm/ReadVariableOp?3layer_normalization_33/batchnorm/mul/ReadVariableOp?6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp?8multi_head_self_attention_9/key/Tensordot/ReadVariableOp?6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp?8multi_head_self_attention_9/out/Tensordot/ReadVariableOp?8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp?:multi_head_self_attention_9/query/Tensordot/ReadVariableOp?8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp?:multi_head_self_attention_9/value/Tensordot/ReadVariableOp?,sequential_9/dense_36/BiasAdd/ReadVariableOp?.sequential_9/dense_36/Tensordot/ReadVariableOp?,sequential_9/dense_37/BiasAdd/ReadVariableOp?.sequential_9/dense_37/Tensordot/ReadVariableOpW
!multi_head_self_attention_9/ShapeShapeinputs*
T0*
_output_shapes
:y
/multi_head_self_attention_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1multi_head_self_attention_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1multi_head_self_attention_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)multi_head_self_attention_9/strided_sliceStridedSlice*multi_head_self_attention_9/Shape:output:08multi_head_self_attention_9/strided_slice/stack:output:0:multi_head_self_attention_9/strided_slice/stack_1:output:0:multi_head_self_attention_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:multi_head_self_attention_9/query/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_9_query_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0z
0multi_head_self_attention_9/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0multi_head_self_attention_9/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
1multi_head_self_attention_9/query/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention_9/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/query/Tensordot/GatherV2GatherV2:multi_head_self_attention_9/query/Tensordot/Shape:output:09multi_head_self_attention_9/query/Tensordot/free:output:0Bmulti_head_self_attention_9/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;multi_head_self_attention_9/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6multi_head_self_attention_9/query/Tensordot/GatherV2_1GatherV2:multi_head_self_attention_9/query/Tensordot/Shape:output:09multi_head_self_attention_9/query/Tensordot/axes:output:0Dmulti_head_self_attention_9/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1multi_head_self_attention_9/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/query/Tensordot/ProdProd=multi_head_self_attention_9/query/Tensordot/GatherV2:output:0:multi_head_self_attention_9/query/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention_9/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2multi_head_self_attention_9/query/Tensordot/Prod_1Prod?multi_head_self_attention_9/query/Tensordot/GatherV2_1:output:0<multi_head_self_attention_9/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention_9/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/query/Tensordot/concatConcatV29multi_head_self_attention_9/query/Tensordot/free:output:09multi_head_self_attention_9/query/Tensordot/axes:output:0@multi_head_self_attention_9/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1multi_head_self_attention_9/query/Tensordot/stackPack9multi_head_self_attention_9/query/Tensordot/Prod:output:0;multi_head_self_attention_9/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5multi_head_self_attention_9/query/Tensordot/transpose	Transposeinputs;multi_head_self_attention_9/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
3multi_head_self_attention_9/query/Tensordot/ReshapeReshape9multi_head_self_attention_9/query/Tensordot/transpose:y:0:multi_head_self_attention_9/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2multi_head_self_attention_9/query/Tensordot/MatMulMatMul<multi_head_self_attention_9/query/Tensordot/Reshape:output:0Bmulti_head_self_attention_9/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
3multi_head_self_attention_9/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:{
9multi_head_self_attention_9/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/query/Tensordot/concat_1ConcatV2=multi_head_self_attention_9/query/Tensordot/GatherV2:output:0<multi_head_self_attention_9/query/Tensordot/Const_2:output:0Bmulti_head_self_attention_9/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+multi_head_self_attention_9/query/TensordotReshape<multi_head_self_attention_9/query/Tensordot/MatMul:product:0=multi_head_self_attention_9/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/query/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_query_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_self_attention_9/query/BiasAddBiasAdd4multi_head_self_attention_9/query/Tensordot:output:0@multi_head_self_attention_9/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/key/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_key_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0x
.multi_head_self_attention_9/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention_9/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_self_attention_9/key/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_self_attention_9/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/key/Tensordot/GatherV2GatherV28multi_head_self_attention_9/key/Tensordot/Shape:output:07multi_head_self_attention_9/key/Tensordot/free:output:0@multi_head_self_attention_9/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention_9/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/key/Tensordot/GatherV2_1GatherV28multi_head_self_attention_9/key/Tensordot/Shape:output:07multi_head_self_attention_9/key/Tensordot/axes:output:0Bmulti_head_self_attention_9/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention_9/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_self_attention_9/key/Tensordot/ProdProd;multi_head_self_attention_9/key/Tensordot/GatherV2:output:08multi_head_self_attention_9/key/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention_9/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/key/Tensordot/Prod_1Prod=multi_head_self_attention_9/key/Tensordot/GatherV2_1:output:0:multi_head_self_attention_9/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention_9/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_self_attention_9/key/Tensordot/concatConcatV27multi_head_self_attention_9/key/Tensordot/free:output:07multi_head_self_attention_9/key/Tensordot/axes:output:0>multi_head_self_attention_9/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_self_attention_9/key/Tensordot/stackPack7multi_head_self_attention_9/key/Tensordot/Prod:output:09multi_head_self_attention_9/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_self_attention_9/key/Tensordot/transpose	Transposeinputs9multi_head_self_attention_9/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
1multi_head_self_attention_9/key/Tensordot/ReshapeReshape7multi_head_self_attention_9/key/Tensordot/transpose:y:08multi_head_self_attention_9/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_self_attention_9/key/Tensordot/MatMulMatMul:multi_head_self_attention_9/key/Tensordot/Reshape:output:0@multi_head_self_attention_9/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_self_attention_9/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_self_attention_9/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/key/Tensordot/concat_1ConcatV2;multi_head_self_attention_9/key/Tensordot/GatherV2:output:0:multi_head_self_attention_9/key/Tensordot/Const_2:output:0@multi_head_self_attention_9/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_self_attention_9/key/TensordotReshape:multi_head_self_attention_9/key/Tensordot/MatMul:product:0;multi_head_self_attention_9/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
6multi_head_self_attention_9/key/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_9_key_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_self_attention_9/key/BiasAddBiasAdd2multi_head_self_attention_9/key/Tensordot:output:0>multi_head_self_attention_9/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
:multi_head_self_attention_9/value/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_9_value_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0z
0multi_head_self_attention_9/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0multi_head_self_attention_9/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
1multi_head_self_attention_9/value/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention_9/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/value/Tensordot/GatherV2GatherV2:multi_head_self_attention_9/value/Tensordot/Shape:output:09multi_head_self_attention_9/value/Tensordot/free:output:0Bmulti_head_self_attention_9/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;multi_head_self_attention_9/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6multi_head_self_attention_9/value/Tensordot/GatherV2_1GatherV2:multi_head_self_attention_9/value/Tensordot/Shape:output:09multi_head_self_attention_9/value/Tensordot/axes:output:0Dmulti_head_self_attention_9/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1multi_head_self_attention_9/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/value/Tensordot/ProdProd=multi_head_self_attention_9/value/Tensordot/GatherV2:output:0:multi_head_self_attention_9/value/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention_9/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2multi_head_self_attention_9/value/Tensordot/Prod_1Prod?multi_head_self_attention_9/value/Tensordot/GatherV2_1:output:0<multi_head_self_attention_9/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention_9/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/value/Tensordot/concatConcatV29multi_head_self_attention_9/value/Tensordot/free:output:09multi_head_self_attention_9/value/Tensordot/axes:output:0@multi_head_self_attention_9/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1multi_head_self_attention_9/value/Tensordot/stackPack9multi_head_self_attention_9/value/Tensordot/Prod:output:0;multi_head_self_attention_9/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5multi_head_self_attention_9/value/Tensordot/transpose	Transposeinputs;multi_head_self_attention_9/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
3multi_head_self_attention_9/value/Tensordot/ReshapeReshape9multi_head_self_attention_9/value/Tensordot/transpose:y:0:multi_head_self_attention_9/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2multi_head_self_attention_9/value/Tensordot/MatMulMatMul<multi_head_self_attention_9/value/Tensordot/Reshape:output:0Bmulti_head_self_attention_9/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
3multi_head_self_attention_9/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:{
9multi_head_self_attention_9/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/value/Tensordot/concat_1ConcatV2=multi_head_self_attention_9/value/Tensordot/GatherV2:output:0<multi_head_self_attention_9/value/Tensordot/Const_2:output:0Bmulti_head_self_attention_9/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+multi_head_self_attention_9/value/TensordotReshape<multi_head_self_attention_9/value/Tensordot/MatMul:product:0=multi_head_self_attention_9/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/value/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_self_attention_9/value/BiasAddBiasAdd4multi_head_self_attention_9/value/Tensordot:output:0@multi_head_self_attention_9/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pm
+multi_head_self_attention_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :pm
+multi_head_self_attention_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention_9/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
)multi_head_self_attention_9/Reshape/shapePack2multi_head_self_attention_9/strided_slice:output:04multi_head_self_attention_9/Reshape/shape/1:output:04multi_head_self_attention_9/Reshape/shape/2:output:04multi_head_self_attention_9/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
#multi_head_self_attention_9/ReshapeReshape2multi_head_self_attention_9/query/BiasAdd:output:02multi_head_self_attention_9/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????p?
*multi_head_self_attention_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
%multi_head_self_attention_9/transpose	Transpose,multi_head_self_attention_9/Reshape:output:03multi_head_self_attention_9/transpose/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-multi_head_self_attention_9/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_1/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_1/shape/1:output:06multi_head_self_attention_9/Reshape_1/shape/2:output:06multi_head_self_attention_9/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_1Reshape0multi_head_self_attention_9/key/BiasAdd:output:04multi_head_self_attention_9/Reshape_1/shape:output:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_1	Transpose.multi_head_self_attention_9/Reshape_1:output:05multi_head_self_attention_9/transpose_1/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-multi_head_self_attention_9/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_2/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_2/shape/1:output:06multi_head_self_attention_9/Reshape_2/shape/2:output:06multi_head_self_attention_9/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_2Reshape2multi_head_self_attention_9/value/BiasAdd:output:04multi_head_self_attention_9/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_2	Transpose.multi_head_self_attention_9/Reshape_2:output:05multi_head_self_attention_9/transpose_2/perm:output:0*
T0*/
_output_shapes
:?????????p?
"multi_head_self_attention_9/MatMulBatchMatMulV2)multi_head_self_attention_9/transpose:y:0+multi_head_self_attention_9/transpose_1:y:0*
T0*/
_output_shapes
:?????????pp*
adj_y(~
#multi_head_self_attention_9/Shape_1Shape+multi_head_self_attention_9/transpose_1:y:0*
T0*
_output_shapes
:?
1multi_head_self_attention_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
3multi_head_self_attention_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3multi_head_self_attention_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+multi_head_self_attention_9/strided_slice_1StridedSlice,multi_head_self_attention_9/Shape_1:output:0:multi_head_self_attention_9/strided_slice_1/stack:output:0<multi_head_self_attention_9/strided_slice_1/stack_1:output:0<multi_head_self_attention_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
 multi_head_self_attention_9/CastCast4multi_head_self_attention_9/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: o
 multi_head_self_attention_9/SqrtSqrt$multi_head_self_attention_9/Cast:y:0*
T0*
_output_shapes
: ?
#multi_head_self_attention_9/truedivRealDiv+multi_head_self_attention_9/MatMul:output:0$multi_head_self_attention_9/Sqrt:y:0*
T0*/
_output_shapes
:?????????pp?
#multi_head_self_attention_9/SoftmaxSoftmax'multi_head_self_attention_9/truediv:z:0*
T0*/
_output_shapes
:?????????pp?
$multi_head_self_attention_9/MatMul_1BatchMatMulV2-multi_head_self_attention_9/Softmax:softmax:0+multi_head_self_attention_9/transpose_2:y:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_3	Transpose-multi_head_self_attention_9/MatMul_1:output:05multi_head_self_attention_9/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_3/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_3/shape/1:output:06multi_head_self_attention_9/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_3Reshape+multi_head_self_attention_9/transpose_3:y:04multi_head_self_attention_9/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/out/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_self_attention_9/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention_9/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
/multi_head_self_attention_9/out/Tensordot/ShapeShape.multi_head_self_attention_9/Reshape_3:output:0*
T0*
_output_shapes
:y
7multi_head_self_attention_9/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/out/Tensordot/GatherV2GatherV28multi_head_self_attention_9/out/Tensordot/Shape:output:07multi_head_self_attention_9/out/Tensordot/free:output:0@multi_head_self_attention_9/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention_9/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/out/Tensordot/GatherV2_1GatherV28multi_head_self_attention_9/out/Tensordot/Shape:output:07multi_head_self_attention_9/out/Tensordot/axes:output:0Bmulti_head_self_attention_9/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention_9/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_self_attention_9/out/Tensordot/ProdProd;multi_head_self_attention_9/out/Tensordot/GatherV2:output:08multi_head_self_attention_9/out/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention_9/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/out/Tensordot/Prod_1Prod=multi_head_self_attention_9/out/Tensordot/GatherV2_1:output:0:multi_head_self_attention_9/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention_9/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_self_attention_9/out/Tensordot/concatConcatV27multi_head_self_attention_9/out/Tensordot/free:output:07multi_head_self_attention_9/out/Tensordot/axes:output:0>multi_head_self_attention_9/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_self_attention_9/out/Tensordot/stackPack7multi_head_self_attention_9/out/Tensordot/Prod:output:09multi_head_self_attention_9/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_self_attention_9/out/Tensordot/transpose	Transpose.multi_head_self_attention_9/Reshape_3:output:09multi_head_self_attention_9/out/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
1multi_head_self_attention_9/out/Tensordot/ReshapeReshape7multi_head_self_attention_9/out/Tensordot/transpose:y:08multi_head_self_attention_9/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_self_attention_9/out/Tensordot/MatMulMatMul:multi_head_self_attention_9/out/Tensordot/Reshape:output:0@multi_head_self_attention_9/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_self_attention_9/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_self_attention_9/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/out/Tensordot/concat_1ConcatV2;multi_head_self_attention_9/out/Tensordot/GatherV2:output:0:multi_head_self_attention_9/out/Tensordot/Const_2:output:0@multi_head_self_attention_9/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_self_attention_9/out/TensordotReshape:multi_head_self_attention_9/out/Tensordot/MatMul:product:0;multi_head_self_attention_9/out/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
6multi_head_self_attention_9/out/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_9_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_self_attention_9/out/BiasAddBiasAdd2multi_head_self_attention_9/out/Tensordot:output:0>multi_head_self_attention_9/out/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
dropout_28/IdentityIdentity0multi_head_self_attention_9/out/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p
5layer_normalization_32/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_32/moments/meanMeandropout_28/Identity:output:0>layer_normalization_32/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
+layer_normalization_32/moments/StopGradientStopGradient,layer_normalization_32/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
0layer_normalization_32/moments/SquaredDifferenceSquaredDifferencedropout_28/Identity:output:04layer_normalization_32/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
9layer_normalization_32/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
'layer_normalization_32/moments/varianceMean4layer_normalization_32/moments/SquaredDifference:z:0Blayer_normalization_32/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(k
&layer_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
$layer_normalization_32/batchnorm/addAddV20layer_normalization_32/moments/variance:output:0/layer_normalization_32/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/RsqrtRsqrt(layer_normalization_32/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
3layer_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_32/batchnorm/mulMul*layer_normalization_32/batchnorm/Rsqrt:y:0;layer_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/mul_1Muldropout_28/Identity:output:0(layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/mul_2Mul,layer_normalization_32/moments/mean:output:0(layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
/layer_normalization_32/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_32/batchnorm/subSub7layer_normalization_32/batchnorm/ReadVariableOp:value:0*layer_normalization_32/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/add_1AddV2*layer_normalization_32/batchnorm/mul_1:z:0(layer_normalization_32/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p?
.sequential_9/dense_36/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_36_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_9/dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_9/dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_9/dense_36/Tensordot/ShapeShape*layer_normalization_32/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_9/dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_36/Tensordot/GatherV2GatherV2.sequential_9/dense_36/Tensordot/Shape:output:0-sequential_9/dense_36/Tensordot/free:output:06sequential_9/dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_9/dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_9/dense_36/Tensordot/GatherV2_1GatherV2.sequential_9/dense_36/Tensordot/Shape:output:0-sequential_9/dense_36/Tensordot/axes:output:08sequential_9/dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_9/dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_9/dense_36/Tensordot/ProdProd1sequential_9/dense_36/Tensordot/GatherV2:output:0.sequential_9/dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_9/dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_9/dense_36/Tensordot/Prod_1Prod3sequential_9/dense_36/Tensordot/GatherV2_1:output:00sequential_9/dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_9/dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_9/dense_36/Tensordot/concatConcatV2-sequential_9/dense_36/Tensordot/free:output:0-sequential_9/dense_36/Tensordot/axes:output:04sequential_9/dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_9/dense_36/Tensordot/stackPack-sequential_9/dense_36/Tensordot/Prod:output:0/sequential_9/dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_9/dense_36/Tensordot/transpose	Transpose*layer_normalization_32/batchnorm/add_1:z:0/sequential_9/dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
'sequential_9/dense_36/Tensordot/ReshapeReshape-sequential_9/dense_36/Tensordot/transpose:y:0.sequential_9/dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_9/dense_36/Tensordot/MatMulMatMul0sequential_9/dense_36/Tensordot/Reshape:output:06sequential_9/dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_9/dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_9/dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_36/Tensordot/concat_1ConcatV21sequential_9/dense_36/Tensordot/GatherV2:output:00sequential_9/dense_36/Tensordot/Const_2:output:06sequential_9/dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_9/dense_36/TensordotReshape0sequential_9/dense_36/Tensordot/MatMul:product:01sequential_9/dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/dense_36/BiasAddBiasAdd(sequential_9/dense_36/Tensordot:output:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
.sequential_9/dense_37/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_37_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_9/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_9/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_9/dense_37/Tensordot/ShapeShape(sequential_9/dense_36/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_9/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_37/Tensordot/GatherV2GatherV2.sequential_9/dense_37/Tensordot/Shape:output:0-sequential_9/dense_37/Tensordot/free:output:06sequential_9/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_9/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_9/dense_37/Tensordot/GatherV2_1GatherV2.sequential_9/dense_37/Tensordot/Shape:output:0-sequential_9/dense_37/Tensordot/axes:output:08sequential_9/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_9/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_9/dense_37/Tensordot/ProdProd1sequential_9/dense_37/Tensordot/GatherV2:output:0.sequential_9/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_9/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_9/dense_37/Tensordot/Prod_1Prod3sequential_9/dense_37/Tensordot/GatherV2_1:output:00sequential_9/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_9/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_9/dense_37/Tensordot/concatConcatV2-sequential_9/dense_37/Tensordot/free:output:0-sequential_9/dense_37/Tensordot/axes:output:04sequential_9/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_9/dense_37/Tensordot/stackPack-sequential_9/dense_37/Tensordot/Prod:output:0/sequential_9/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_9/dense_37/Tensordot/transpose	Transpose(sequential_9/dense_36/Relu:activations:0/sequential_9/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
'sequential_9/dense_37/Tensordot/ReshapeReshape-sequential_9/dense_37/Tensordot/transpose:y:0.sequential_9/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_9/dense_37/Tensordot/MatMulMatMul0sequential_9/dense_37/Tensordot/Reshape:output:06sequential_9/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_9/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_9/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_37/Tensordot/concat_1ConcatV21sequential_9/dense_37/Tensordot/GatherV2:output:00sequential_9/dense_37/Tensordot/Const_2:output:06sequential_9/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_9/dense_37/TensordotReshape0sequential_9/dense_37/Tensordot/MatMul:product:01sequential_9/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/dense_37/BiasAddBiasAdd(sequential_9/dense_37/Tensordot:output:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p}
dropout_29/IdentityIdentity&sequential_9/dense_37/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
addAddV2*layer_normalization_32/batchnorm/add_1:z:0dropout_29/Identity:output:0*
T0*+
_output_shapes
:?????????p
5layer_normalization_33/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_33/moments/meanMeanadd:z:0>layer_normalization_33/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
+layer_normalization_33/moments/StopGradientStopGradient,layer_normalization_33/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
0layer_normalization_33/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_33/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
9layer_normalization_33/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
'layer_normalization_33/moments/varianceMean4layer_normalization_33/moments/SquaredDifference:z:0Blayer_normalization_33/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(k
&layer_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
$layer_normalization_33/batchnorm/addAddV20layer_normalization_33/moments/variance:output:0/layer_normalization_33/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/RsqrtRsqrt(layer_normalization_33/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
3layer_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_33/batchnorm/mulMul*layer_normalization_33/batchnorm/Rsqrt:y:0;layer_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/mul_1Muladd:z:0(layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/mul_2Mul,layer_normalization_33/moments/mean:output:0(layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
/layer_normalization_33/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_33/batchnorm/subSub7layer_normalization_33/batchnorm/ReadVariableOp:value:0*layer_normalization_33/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/add_1AddV2*layer_normalization_33/batchnorm/mul_1:z:0(layer_normalization_33/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p}
IdentityIdentity*layer_normalization_33/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp0^layer_normalization_32/batchnorm/ReadVariableOp4^layer_normalization_32/batchnorm/mul/ReadVariableOp0^layer_normalization_33/batchnorm/ReadVariableOp4^layer_normalization_33/batchnorm/mul/ReadVariableOp7^multi_head_self_attention_9/key/BiasAdd/ReadVariableOp9^multi_head_self_attention_9/key/Tensordot/ReadVariableOp7^multi_head_self_attention_9/out/BiasAdd/ReadVariableOp9^multi_head_self_attention_9/out/Tensordot/ReadVariableOp9^multi_head_self_attention_9/query/BiasAdd/ReadVariableOp;^multi_head_self_attention_9/query/Tensordot/ReadVariableOp9^multi_head_self_attention_9/value/BiasAdd/ReadVariableOp;^multi_head_self_attention_9/value/Tensordot/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp/^sequential_9/dense_36/Tensordot/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp/^sequential_9/dense_37/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????p@: : : : : : : : : : : : : : : : 2b
/layer_normalization_32/batchnorm/ReadVariableOp/layer_normalization_32/batchnorm/ReadVariableOp2j
3layer_normalization_32/batchnorm/mul/ReadVariableOp3layer_normalization_32/batchnorm/mul/ReadVariableOp2b
/layer_normalization_33/batchnorm/ReadVariableOp/layer_normalization_33/batchnorm/ReadVariableOp2j
3layer_normalization_33/batchnorm/mul/ReadVariableOp3layer_normalization_33/batchnorm/mul/ReadVariableOp2p
6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp2t
8multi_head_self_attention_9/key/Tensordot/ReadVariableOp8multi_head_self_attention_9/key/Tensordot/ReadVariableOp2p
6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp2t
8multi_head_self_attention_9/out/Tensordot/ReadVariableOp8multi_head_self_attention_9/out/Tensordot/ReadVariableOp2t
8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp2x
:multi_head_self_attention_9/query/Tensordot/ReadVariableOp:multi_head_self_attention_9/query/Tensordot/ReadVariableOp2t
8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp2x
:multi_head_self_attention_9/value/Tensordot/ReadVariableOp:multi_head_self_attention_9/value/Tensordot/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2`
.sequential_9/dense_36/Tensordot/ReadVariableOp.sequential_9/dense_36/Tensordot/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2`
.sequential_9/dense_37/Tensordot/ReadVariableOp.sequential_9/dense_37/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
G
+__inference_dropout_27_layer_call_fn_258547

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_256122d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????p@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????p@:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
?
R__inference_layer_normalization_34_layer_call_and_return_conditional_losses_259320

inputs+
mul_2_readvariableop_resource:)
add_readvariableop_resource:
identity??add/ReadVariableOp?mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_dense_39_layer_call_and_return_conditional_losses_259352

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_39/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_39/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_39/kernel/Regularizer/Square/ReadVariableOp1dense_39/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_30_layer_call_and_return_conditional_losses_259225

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?!
!__inference__wrapped_model_255874
input_10D
2model_4_dense_35_tensordot_readvariableop_resource:d@>
0model_4_dense_35_biasadd_readvariableop_resource:@J
<model_4_layer_normalization_31_mul_3_readvariableop_resource:@H
:model_4_layer_normalization_31_add_readvariableop_resource:@q
_model_4_transformer_block_9_multi_head_self_attention_9_query_tensordot_readvariableop_resource:@k
]model_4_transformer_block_9_multi_head_self_attention_9_query_biasadd_readvariableop_resource:o
]model_4_transformer_block_9_multi_head_self_attention_9_key_tensordot_readvariableop_resource:@i
[model_4_transformer_block_9_multi_head_self_attention_9_key_biasadd_readvariableop_resource:q
_model_4_transformer_block_9_multi_head_self_attention_9_value_tensordot_readvariableop_resource:@k
]model_4_transformer_block_9_multi_head_self_attention_9_value_biasadd_readvariableop_resource:o
]model_4_transformer_block_9_multi_head_self_attention_9_out_tensordot_readvariableop_resource:i
[model_4_transformer_block_9_multi_head_self_attention_9_out_biasadd_readvariableop_resource:f
Xmodel_4_transformer_block_9_layer_normalization_32_batchnorm_mul_readvariableop_resource:b
Tmodel_4_transformer_block_9_layer_normalization_32_batchnorm_readvariableop_resource:e
Smodel_4_transformer_block_9_sequential_9_dense_36_tensordot_readvariableop_resource:_
Qmodel_4_transformer_block_9_sequential_9_dense_36_biasadd_readvariableop_resource:e
Smodel_4_transformer_block_9_sequential_9_dense_37_tensordot_readvariableop_resource:_
Qmodel_4_transformer_block_9_sequential_9_dense_37_biasadd_readvariableop_resource:f
Xmodel_4_transformer_block_9_layer_normalization_33_batchnorm_mul_readvariableop_resource:b
Tmodel_4_transformer_block_9_layer_normalization_33_batchnorm_readvariableop_resource:B
/model_4_dense_38_matmul_readvariableop_resource:	?>
0model_4_dense_38_biasadd_readvariableop_resource:J
<model_4_layer_normalization_34_mul_2_readvariableop_resource:H
:model_4_layer_normalization_34_add_readvariableop_resource:A
/model_4_dense_39_matmul_readvariableop_resource:>
0model_4_dense_39_biasadd_readvariableop_resource:
identity??'model_4/dense_35/BiasAdd/ReadVariableOp?)model_4/dense_35/Tensordot/ReadVariableOp?'model_4/dense_38/BiasAdd/ReadVariableOp?&model_4/dense_38/MatMul/ReadVariableOp?'model_4/dense_39/BiasAdd/ReadVariableOp?&model_4/dense_39/MatMul/ReadVariableOp?1model_4/layer_normalization_31/add/ReadVariableOp?3model_4/layer_normalization_31/mul_3/ReadVariableOp?1model_4/layer_normalization_34/add/ReadVariableOp?3model_4/layer_normalization_34/mul_2/ReadVariableOp?Kmodel_4/transformer_block_9/layer_normalization_32/batchnorm/ReadVariableOp?Omodel_4/transformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOp?Kmodel_4/transformer_block_9/layer_normalization_33/batchnorm/ReadVariableOp?Omodel_4/transformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOp?Rmodel_4/transformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOp?Tmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOp?Rmodel_4/transformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOp?Tmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOp?Tmodel_4/transformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOp?Vmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOp?Tmodel_4/transformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOp?Vmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOp?Hmodel_4/transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOp?Jmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOp?Hmodel_4/transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOp?Jmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp?
)model_4/dense_35/Tensordot/ReadVariableOpReadVariableOp2model_4_dense_35_tensordot_readvariableop_resource*
_output_shapes

:d@*
dtype0i
model_4/dense_35/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
model_4/dense_35/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       X
 model_4/dense_35/Tensordot/ShapeShapeinput_10*
T0*
_output_shapes
:j
(model_4/dense_35/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#model_4/dense_35/Tensordot/GatherV2GatherV2)model_4/dense_35/Tensordot/Shape:output:0(model_4/dense_35/Tensordot/free:output:01model_4/dense_35/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*model_4/dense_35/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%model_4/dense_35/Tensordot/GatherV2_1GatherV2)model_4/dense_35/Tensordot/Shape:output:0(model_4/dense_35/Tensordot/axes:output:03model_4/dense_35/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 model_4/dense_35/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
model_4/dense_35/Tensordot/ProdProd,model_4/dense_35/Tensordot/GatherV2:output:0)model_4/dense_35/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"model_4/dense_35/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
!model_4/dense_35/Tensordot/Prod_1Prod.model_4/dense_35/Tensordot/GatherV2_1:output:0+model_4/dense_35/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&model_4/dense_35/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!model_4/dense_35/Tensordot/concatConcatV2(model_4/dense_35/Tensordot/free:output:0(model_4/dense_35/Tensordot/axes:output:0/model_4/dense_35/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 model_4/dense_35/Tensordot/stackPack(model_4/dense_35/Tensordot/Prod:output:0*model_4/dense_35/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
$model_4/dense_35/Tensordot/transpose	Transposeinput_10*model_4/dense_35/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????pd?
"model_4/dense_35/Tensordot/ReshapeReshape(model_4/dense_35/Tensordot/transpose:y:0)model_4/dense_35/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
!model_4/dense_35/Tensordot/MatMulMatMul+model_4/dense_35/Tensordot/Reshape:output:01model_4/dense_35/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@l
"model_4/dense_35/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@j
(model_4/dense_35/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#model_4/dense_35/Tensordot/concat_1ConcatV2,model_4/dense_35/Tensordot/GatherV2:output:0+model_4/dense_35/Tensordot/Const_2:output:01model_4/dense_35/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
model_4/dense_35/TensordotReshape+model_4/dense_35/Tensordot/MatMul:product:0,model_4/dense_35/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p@?
'model_4/dense_35/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_4/dense_35/BiasAddBiasAdd#model_4/dense_35/Tensordot:output:0/model_4/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@v
model_4/dense_35/ReluRelu!model_4/dense_35/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p@?
model_4/dropout_27/IdentityIdentity#model_4/dense_35/Relu:activations:0*
T0*+
_output_shapes
:?????????p@x
$model_4/layer_normalization_31/ShapeShape$model_4/dropout_27/Identity:output:0*
T0*
_output_shapes
:|
2model_4/layer_normalization_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4model_4/layer_normalization_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4model_4/layer_normalization_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,model_4/layer_normalization_31/strided_sliceStridedSlice-model_4/layer_normalization_31/Shape:output:0;model_4/layer_normalization_31/strided_slice/stack:output:0=model_4/layer_normalization_31/strided_slice/stack_1:output:0=model_4/layer_normalization_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$model_4/layer_normalization_31/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
"model_4/layer_normalization_31/mulMul-model_4/layer_normalization_31/mul/x:output:05model_4/layer_normalization_31/strided_slice:output:0*
T0*
_output_shapes
: ~
4model_4/layer_normalization_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
6model_4/layer_normalization_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6model_4/layer_normalization_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.model_4/layer_normalization_31/strided_slice_1StridedSlice-model_4/layer_normalization_31/Shape:output:0=model_4/layer_normalization_31/strided_slice_1/stack:output:0?model_4/layer_normalization_31/strided_slice_1/stack_1:output:0?model_4/layer_normalization_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
$model_4/layer_normalization_31/mul_1Mul&model_4/layer_normalization_31/mul:z:07model_4/layer_normalization_31/strided_slice_1:output:0*
T0*
_output_shapes
: ~
4model_4/layer_normalization_31/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
6model_4/layer_normalization_31/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6model_4/layer_normalization_31/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.model_4/layer_normalization_31/strided_slice_2StridedSlice-model_4/layer_normalization_31/Shape:output:0=model_4/layer_normalization_31/strided_slice_2/stack:output:0?model_4/layer_normalization_31/strided_slice_2/stack_1:output:0?model_4/layer_normalization_31/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model_4/layer_normalization_31/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
$model_4/layer_normalization_31/mul_2Mul/model_4/layer_normalization_31/mul_2/x:output:07model_4/layer_normalization_31/strided_slice_2:output:0*
T0*
_output_shapes
: p
.model_4/layer_normalization_31/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :p
.model_4/layer_normalization_31/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
,model_4/layer_normalization_31/Reshape/shapePack7model_4/layer_normalization_31/Reshape/shape/0:output:0(model_4/layer_normalization_31/mul_1:z:0(model_4/layer_normalization_31/mul_2:z:07model_4/layer_normalization_31/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
&model_4/layer_normalization_31/ReshapeReshape$model_4/dropout_27/Identity:output:05model_4/layer_normalization_31/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@?
*model_4/layer_normalization_31/ones/packedPack(model_4/layer_normalization_31/mul_1:z:0*
N*
T0*
_output_shapes
:n
)model_4/layer_normalization_31/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#model_4/layer_normalization_31/onesFill3model_4/layer_normalization_31/ones/packed:output:02model_4/layer_normalization_31/ones/Const:output:0*
T0*#
_output_shapes
:??????????
+model_4/layer_normalization_31/zeros/packedPack(model_4/layer_normalization_31/mul_1:z:0*
N*
T0*
_output_shapes
:o
*model_4/layer_normalization_31/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
$model_4/layer_normalization_31/zerosFill4model_4/layer_normalization_31/zeros/packed:output:03model_4/layer_normalization_31/zeros/Const:output:0*
T0*#
_output_shapes
:?????????g
$model_4/layer_normalization_31/ConstConst*
_output_shapes
: *
dtype0*
valueB i
&model_4/layer_normalization_31/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
/model_4/layer_normalization_31/FusedBatchNormV3FusedBatchNormV3/model_4/layer_normalization_31/Reshape:output:0,model_4/layer_normalization_31/ones:output:0-model_4/layer_normalization_31/zeros:output:0-model_4/layer_normalization_31/Const:output:0/model_4/layer_normalization_31/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
(model_4/layer_normalization_31/Reshape_1Reshape3model_4/layer_normalization_31/FusedBatchNormV3:y:0-model_4/layer_normalization_31/Shape:output:0*
T0*+
_output_shapes
:?????????p@?
3model_4/layer_normalization_31/mul_3/ReadVariableOpReadVariableOp<model_4_layer_normalization_31_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0?
$model_4/layer_normalization_31/mul_3Mul1model_4/layer_normalization_31/Reshape_1:output:0;model_4/layer_normalization_31/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@?
1model_4/layer_normalization_31/add/ReadVariableOpReadVariableOp:model_4_layer_normalization_31_add_readvariableop_resource*
_output_shapes
:@*
dtype0?
"model_4/layer_normalization_31/addAddV2(model_4/layer_normalization_31/mul_3:z:09model_4/layer_normalization_31/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@?
=model_4/transformer_block_9/multi_head_self_attention_9/ShapeShape&model_4/layer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Kmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Emodel_4/transformer_block_9/multi_head_self_attention_9/strided_sliceStridedSliceFmodel_4/transformer_block_9/multi_head_self_attention_9/Shape:output:0Tmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice/stack:output:0Vmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice/stack_1:output:0Vmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Vmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOpReadVariableOp_model_4_transformer_block_9_multi_head_self_attention_9_query_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0?
Lmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/ShapeShape&model_4/layer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Umodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Pmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2GatherV2Vmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/Shape:output:0Umodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/free:output:0^model_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Wmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Rmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1GatherV2Vmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/Shape:output:0Umodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/axes:output:0`model_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Lmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/ProdProdYmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2:output:0Vmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Omodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Nmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/Prod_1Prod[model_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1:output:0Xmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Smodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/concatConcatV2Umodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/free:output:0Umodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/axes:output:0\model_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/stackPackUmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/Prod:output:0Wmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Qmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/transpose	Transpose&model_4/layer_normalization_31/add:z:0Wmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
Omodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/ReshapeReshapeUmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/transpose:y:0Vmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Nmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/MatMulMatMulXmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/Reshape:output:0^model_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Omodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Umodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Pmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1ConcatV2Ymodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2:output:0Xmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_2:output:0^model_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Gmodel_4/transformer_block_9/multi_head_self_attention_9/query/TensordotReshapeXmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/MatMul:product:0Ymodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Tmodel_4/transformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOpReadVariableOp]model_4_transformer_block_9_multi_head_self_attention_9_query_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Emodel_4/transformer_block_9/multi_head_self_attention_9/query/BiasAddBiasAddPmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot:output:0\model_4/transformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
Tmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOpReadVariableOp]model_4_transformer_block_9_multi_head_self_attention_9_key_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0?
Jmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Kmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/ShapeShape&model_4/layer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Smodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2GatherV2Tmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/Shape:output:0Smodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/free:output:0\model_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Umodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Pmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1GatherV2Tmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/Shape:output:0Smodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/axes:output:0^model_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Kmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Jmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/ProdProdWmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2:output:0Tmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Lmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/Prod_1ProdYmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1:output:0Vmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Qmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Lmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/concatConcatV2Smodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/free:output:0Smodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/axes:output:0Zmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Kmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/stackPackSmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/Prod:output:0Umodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Omodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/transpose	Transpose&model_4/layer_normalization_31/add:z:0Umodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/ReshapeReshapeSmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/transpose:y:0Tmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Lmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/MatMulMatMulVmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/Reshape:output:0\model_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Mmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Smodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1ConcatV2Wmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2:output:0Vmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_2:output:0\model_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Emodel_4/transformer_block_9/multi_head_self_attention_9/key/TensordotReshapeVmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/MatMul:product:0Wmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Rmodel_4/transformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOpReadVariableOp[model_4_transformer_block_9_multi_head_self_attention_9_key_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Cmodel_4/transformer_block_9/multi_head_self_attention_9/key/BiasAddBiasAddNmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot:output:0Zmodel_4/transformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
Vmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOpReadVariableOp_model_4_transformer_block_9_multi_head_self_attention_9_value_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0?
Lmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/ShapeShape&model_4/layer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Umodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Pmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2GatherV2Vmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/Shape:output:0Umodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/free:output:0^model_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Wmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Rmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1GatherV2Vmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/Shape:output:0Umodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/axes:output:0`model_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Lmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/ProdProdYmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2:output:0Vmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Omodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Nmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/Prod_1Prod[model_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1:output:0Xmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Smodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/concatConcatV2Umodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/free:output:0Umodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/axes:output:0\model_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/stackPackUmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/Prod:output:0Wmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Qmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/transpose	Transpose&model_4/layer_normalization_31/add:z:0Wmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
Omodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/ReshapeReshapeUmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/transpose:y:0Vmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Nmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/MatMulMatMulXmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/Reshape:output:0^model_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Omodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Umodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Pmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1ConcatV2Ymodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2:output:0Xmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_2:output:0^model_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Gmodel_4/transformer_block_9/multi_head_self_attention_9/value/TensordotReshapeXmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/MatMul:product:0Ymodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Tmodel_4/transformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOpReadVariableOp]model_4_transformer_block_9_multi_head_self_attention_9_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Emodel_4/transformer_block_9/multi_head_self_attention_9/value/BiasAddBiasAddPmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot:output:0\model_4/transformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
Gmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
Gmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Gmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Emodel_4/transformer_block_9/multi_head_self_attention_9/Reshape/shapePackNmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice:output:0Pmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape/shape/1:output:0Pmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape/shape/2:output:0Pmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
?model_4/transformer_block_9/multi_head_self_attention_9/ReshapeReshapeNmodel_4/transformer_block_9/multi_head_self_attention_9/query/BiasAdd:output:0Nmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????p?
Fmodel_4/transformer_block_9/multi_head_self_attention_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Amodel_4/transformer_block_9/multi_head_self_attention_9/transpose	TransposeHmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape:output:0Omodel_4/transformer_block_9/multi_head_self_attention_9/transpose/perm:output:0*
T0*/
_output_shapes
:?????????p?
Imodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
Imodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Imodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Gmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_1/shapePackNmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice:output:0Rmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_1/shape/1:output:0Rmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_1/shape/2:output:0Rmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
Amodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_1ReshapeLmodel_4/transformer_block_9/multi_head_self_attention_9/key/BiasAdd:output:0Pmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_1/shape:output:0*
T0*/
_output_shapes
:?????????p?
Hmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Cmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_1	TransposeJmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_1:output:0Qmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_1/perm:output:0*
T0*/
_output_shapes
:?????????p?
Imodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
Imodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Imodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Gmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_2/shapePackNmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice:output:0Rmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_2/shape/1:output:0Rmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_2/shape/2:output:0Rmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
Amodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_2ReshapeNmodel_4/transformer_block_9/multi_head_self_attention_9/value/BiasAdd:output:0Pmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????p?
Hmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Cmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_2	TransposeJmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_2:output:0Qmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_2/perm:output:0*
T0*/
_output_shapes
:?????????p?
>model_4/transformer_block_9/multi_head_self_attention_9/MatMulBatchMatMulV2Emodel_4/transformer_block_9/multi_head_self_attention_9/transpose:y:0Gmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_1:y:0*
T0*/
_output_shapes
:?????????pp*
adj_y(?
?model_4/transformer_block_9/multi_head_self_attention_9/Shape_1ShapeGmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_1:y:0*
T0*
_output_shapes
:?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Omodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Omodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Gmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice_1StridedSliceHmodel_4/transformer_block_9/multi_head_self_attention_9/Shape_1:output:0Vmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice_1/stack:output:0Xmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_1:output:0Xmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<model_4/transformer_block_9/multi_head_self_attention_9/CastCastPmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
<model_4/transformer_block_9/multi_head_self_attention_9/SqrtSqrt@model_4/transformer_block_9/multi_head_self_attention_9/Cast:y:0*
T0*
_output_shapes
: ?
?model_4/transformer_block_9/multi_head_self_attention_9/truedivRealDivGmodel_4/transformer_block_9/multi_head_self_attention_9/MatMul:output:0@model_4/transformer_block_9/multi_head_self_attention_9/Sqrt:y:0*
T0*/
_output_shapes
:?????????pp?
?model_4/transformer_block_9/multi_head_self_attention_9/SoftmaxSoftmaxCmodel_4/transformer_block_9/multi_head_self_attention_9/truediv:z:0*
T0*/
_output_shapes
:?????????pp?
@model_4/transformer_block_9/multi_head_self_attention_9/MatMul_1BatchMatMulV2Imodel_4/transformer_block_9/multi_head_self_attention_9/Softmax:softmax:0Gmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_2:y:0*
T0*/
_output_shapes
:?????????p?
Hmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Cmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_3	TransposeImodel_4/transformer_block_9/multi_head_self_attention_9/MatMul_1:output:0Qmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????p?
Imodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
Imodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Gmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_3/shapePackNmodel_4/transformer_block_9/multi_head_self_attention_9/strided_slice:output:0Rmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_3/shape/1:output:0Rmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
Amodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_3ReshapeGmodel_4/transformer_block_9/multi_head_self_attention_9/transpose_3:y:0Pmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????p?
Tmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOpReadVariableOp]model_4_transformer_block_9_multi_head_self_attention_9_out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Jmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Kmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/ShapeShapeJmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_3:output:0*
T0*
_output_shapes
:?
Smodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2GatherV2Tmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/Shape:output:0Smodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/free:output:0\model_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Umodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Pmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1GatherV2Tmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/Shape:output:0Smodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/axes:output:0^model_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Kmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Jmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/ProdProdWmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2:output:0Tmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Lmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/Prod_1ProdYmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1:output:0Vmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Qmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Lmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/concatConcatV2Smodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/free:output:0Smodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/axes:output:0Zmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Kmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/stackPackSmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/Prod:output:0Umodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Omodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/transpose	TransposeJmodel_4/transformer_block_9/multi_head_self_attention_9/Reshape_3:output:0Umodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
Mmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/ReshapeReshapeSmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/transpose:y:0Tmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Lmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/MatMulMatMulVmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/Reshape:output:0\model_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Mmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Smodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1ConcatV2Wmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2:output:0Vmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_2:output:0\model_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Emodel_4/transformer_block_9/multi_head_self_attention_9/out/TensordotReshapeVmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/MatMul:product:0Wmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Rmodel_4/transformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOpReadVariableOp[model_4_transformer_block_9_multi_head_self_attention_9_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Cmodel_4/transformer_block_9/multi_head_self_attention_9/out/BiasAddBiasAddNmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot:output:0Zmodel_4/transformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
/model_4/transformer_block_9/dropout_28/IdentityIdentityLmodel_4/transformer_block_9/multi_head_self_attention_9/out/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
Qmodel_4/transformer_block_9/layer_normalization_32/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
?model_4/transformer_block_9/layer_normalization_32/moments/meanMean8model_4/transformer_block_9/dropout_28/Identity:output:0Zmodel_4/transformer_block_9/layer_normalization_32/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
Gmodel_4/transformer_block_9/layer_normalization_32/moments/StopGradientStopGradientHmodel_4/transformer_block_9/layer_normalization_32/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
Lmodel_4/transformer_block_9/layer_normalization_32/moments/SquaredDifferenceSquaredDifference8model_4/transformer_block_9/dropout_28/Identity:output:0Pmodel_4/transformer_block_9/layer_normalization_32/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
Umodel_4/transformer_block_9/layer_normalization_32/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Cmodel_4/transformer_block_9/layer_normalization_32/moments/varianceMeanPmodel_4/transformer_block_9/layer_normalization_32/moments/SquaredDifference:z:0^model_4/transformer_block_9/layer_normalization_32/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
Bmodel_4/transformer_block_9/layer_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
@model_4/transformer_block_9/layer_normalization_32/batchnorm/addAddV2Lmodel_4/transformer_block_9/layer_normalization_32/moments/variance:output:0Kmodel_4/transformer_block_9/layer_normalization_32/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
Bmodel_4/transformer_block_9/layer_normalization_32/batchnorm/RsqrtRsqrtDmodel_4/transformer_block_9/layer_normalization_32/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
Omodel_4/transformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_4_transformer_block_9_layer_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
@model_4/transformer_block_9/layer_normalization_32/batchnorm/mulMulFmodel_4/transformer_block_9/layer_normalization_32/batchnorm/Rsqrt:y:0Wmodel_4/transformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
Bmodel_4/transformer_block_9/layer_normalization_32/batchnorm/mul_1Mul8model_4/transformer_block_9/dropout_28/Identity:output:0Dmodel_4/transformer_block_9/layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
Bmodel_4/transformer_block_9/layer_normalization_32/batchnorm/mul_2MulHmodel_4/transformer_block_9/layer_normalization_32/moments/mean:output:0Dmodel_4/transformer_block_9/layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
Kmodel_4/transformer_block_9/layer_normalization_32/batchnorm/ReadVariableOpReadVariableOpTmodel_4_transformer_block_9_layer_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
@model_4/transformer_block_9/layer_normalization_32/batchnorm/subSubSmodel_4/transformer_block_9/layer_normalization_32/batchnorm/ReadVariableOp:value:0Fmodel_4/transformer_block_9/layer_normalization_32/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
Bmodel_4/transformer_block_9/layer_normalization_32/batchnorm/add_1AddV2Fmodel_4/transformer_block_9/layer_normalization_32/batchnorm/mul_1:z:0Dmodel_4/transformer_block_9/layer_normalization_32/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p?
Jmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOpReadVariableOpSmodel_4_transformer_block_9_sequential_9_dense_36_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
@model_4/transformer_block_9/sequential_9/dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
@model_4/transformer_block_9/sequential_9/dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Amodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/ShapeShapeFmodel_4/transformer_block_9/layer_normalization_32/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
Imodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2GatherV2Jmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/Shape:output:0Imodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/free:output:0Rmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Kmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Fmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1GatherV2Jmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/Shape:output:0Imodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/axes:output:0Tmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Amodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
@model_4/transformer_block_9/sequential_9/dense_36/Tensordot/ProdProdMmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2:output:0Jmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Cmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Bmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/Prod_1ProdOmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1:output:0Lmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Gmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/concatConcatV2Imodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/free:output:0Imodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/axes:output:0Pmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Amodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/stackPackImodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/Prod:output:0Kmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Emodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/transpose	TransposeFmodel_4/transformer_block_9/layer_normalization_32/batchnorm/add_1:z:0Kmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
Cmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/ReshapeReshapeImodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/transpose:y:0Jmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Bmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/MatMulMatMulLmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/Reshape:output:0Rmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Cmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Imodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/concat_1ConcatV2Mmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2:output:0Lmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/Const_2:output:0Rmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
;model_4/transformer_block_9/sequential_9/dense_36/TensordotReshapeLmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/MatMul:product:0Mmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Hmodel_4/transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOpQmodel_4_transformer_block_9_sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
9model_4/transformer_block_9/sequential_9/dense_36/BiasAddBiasAddDmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot:output:0Pmodel_4/transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
6model_4/transformer_block_9/sequential_9/dense_36/ReluReluBmodel_4/transformer_block_9/sequential_9/dense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
Jmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOpReadVariableOpSmodel_4_transformer_block_9_sequential_9_dense_37_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
@model_4/transformer_block_9/sequential_9/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
@model_4/transformer_block_9/sequential_9/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Amodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/ShapeShapeDmodel_4/transformer_block_9/sequential_9/dense_36/Relu:activations:0*
T0*
_output_shapes
:?
Imodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2GatherV2Jmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/Shape:output:0Imodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/free:output:0Rmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Kmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Fmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1GatherV2Jmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/Shape:output:0Imodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/axes:output:0Tmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Amodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
@model_4/transformer_block_9/sequential_9/dense_37/Tensordot/ProdProdMmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2:output:0Jmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Cmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Bmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/Prod_1ProdOmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1:output:0Lmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Gmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/concatConcatV2Imodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/free:output:0Imodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/axes:output:0Pmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Amodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/stackPackImodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/Prod:output:0Kmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Emodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/transpose	TransposeDmodel_4/transformer_block_9/sequential_9/dense_36/Relu:activations:0Kmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
Cmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/ReshapeReshapeImodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/transpose:y:0Jmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Bmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/MatMulMatMulLmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/Reshape:output:0Rmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Cmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Imodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/concat_1ConcatV2Mmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2:output:0Lmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/Const_2:output:0Rmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
;model_4/transformer_block_9/sequential_9/dense_37/TensordotReshapeLmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/MatMul:product:0Mmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Hmodel_4/transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOpQmodel_4_transformer_block_9_sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
9model_4/transformer_block_9/sequential_9/dense_37/BiasAddBiasAddDmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot:output:0Pmodel_4/transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
/model_4/transformer_block_9/dropout_29/IdentityIdentityBmodel_4/transformer_block_9/sequential_9/dense_37/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
model_4/transformer_block_9/addAddV2Fmodel_4/transformer_block_9/layer_normalization_32/batchnorm/add_1:z:08model_4/transformer_block_9/dropout_29/Identity:output:0*
T0*+
_output_shapes
:?????????p?
Qmodel_4/transformer_block_9/layer_normalization_33/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
?model_4/transformer_block_9/layer_normalization_33/moments/meanMean#model_4/transformer_block_9/add:z:0Zmodel_4/transformer_block_9/layer_normalization_33/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
Gmodel_4/transformer_block_9/layer_normalization_33/moments/StopGradientStopGradientHmodel_4/transformer_block_9/layer_normalization_33/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
Lmodel_4/transformer_block_9/layer_normalization_33/moments/SquaredDifferenceSquaredDifference#model_4/transformer_block_9/add:z:0Pmodel_4/transformer_block_9/layer_normalization_33/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
Umodel_4/transformer_block_9/layer_normalization_33/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Cmodel_4/transformer_block_9/layer_normalization_33/moments/varianceMeanPmodel_4/transformer_block_9/layer_normalization_33/moments/SquaredDifference:z:0^model_4/transformer_block_9/layer_normalization_33/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
Bmodel_4/transformer_block_9/layer_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
@model_4/transformer_block_9/layer_normalization_33/batchnorm/addAddV2Lmodel_4/transformer_block_9/layer_normalization_33/moments/variance:output:0Kmodel_4/transformer_block_9/layer_normalization_33/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
Bmodel_4/transformer_block_9/layer_normalization_33/batchnorm/RsqrtRsqrtDmodel_4/transformer_block_9/layer_normalization_33/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
Omodel_4/transformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_4_transformer_block_9_layer_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
@model_4/transformer_block_9/layer_normalization_33/batchnorm/mulMulFmodel_4/transformer_block_9/layer_normalization_33/batchnorm/Rsqrt:y:0Wmodel_4/transformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
Bmodel_4/transformer_block_9/layer_normalization_33/batchnorm/mul_1Mul#model_4/transformer_block_9/add:z:0Dmodel_4/transformer_block_9/layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
Bmodel_4/transformer_block_9/layer_normalization_33/batchnorm/mul_2MulHmodel_4/transformer_block_9/layer_normalization_33/moments/mean:output:0Dmodel_4/transformer_block_9/layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
Kmodel_4/transformer_block_9/layer_normalization_33/batchnorm/ReadVariableOpReadVariableOpTmodel_4_transformer_block_9_layer_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
@model_4/transformer_block_9/layer_normalization_33/batchnorm/subSubSmodel_4/transformer_block_9/layer_normalization_33/batchnorm/ReadVariableOp:value:0Fmodel_4/transformer_block_9/layer_normalization_33/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
Bmodel_4/transformer_block_9/layer_normalization_33/batchnorm/add_1AddV2Fmodel_4/transformer_block_9/layer_normalization_33/batchnorm/mul_1:z:0Dmodel_4/transformer_block_9/layer_normalization_33/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????ph
model_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
model_4/flatten_4/ReshapeReshapeFmodel_4/transformer_block_9/layer_normalization_33/batchnorm/add_1:z:0 model_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????~
model_4/dropout_30/IdentityIdentity"model_4/flatten_4/Reshape:output:0*
T0*(
_output_shapes
:???????????
&model_4/dense_38/MatMul/ReadVariableOpReadVariableOp/model_4_dense_38_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_4/dense_38/MatMulMatMul$model_4/dropout_30/Identity:output:0.model_4/dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'model_4/dense_38/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_4/dense_38/BiasAddBiasAdd!model_4/dense_38/MatMul:product:0/model_4/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
model_4/dense_38/ReluRelu!model_4/dense_38/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
$model_4/layer_normalization_34/ShapeShape#model_4/dense_38/Relu:activations:0*
T0*
_output_shapes
:|
2model_4/layer_normalization_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4model_4/layer_normalization_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4model_4/layer_normalization_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,model_4/layer_normalization_34/strided_sliceStridedSlice-model_4/layer_normalization_34/Shape:output:0;model_4/layer_normalization_34/strided_slice/stack:output:0=model_4/layer_normalization_34/strided_slice/stack_1:output:0=model_4/layer_normalization_34/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$model_4/layer_normalization_34/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
"model_4/layer_normalization_34/mulMul-model_4/layer_normalization_34/mul/x:output:05model_4/layer_normalization_34/strided_slice:output:0*
T0*
_output_shapes
: ~
4model_4/layer_normalization_34/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
6model_4/layer_normalization_34/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6model_4/layer_normalization_34/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.model_4/layer_normalization_34/strided_slice_1StridedSlice-model_4/layer_normalization_34/Shape:output:0=model_4/layer_normalization_34/strided_slice_1/stack:output:0?model_4/layer_normalization_34/strided_slice_1/stack_1:output:0?model_4/layer_normalization_34/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model_4/layer_normalization_34/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :?
$model_4/layer_normalization_34/mul_1Mul/model_4/layer_normalization_34/mul_1/x:output:07model_4/layer_normalization_34/strided_slice_1:output:0*
T0*
_output_shapes
: p
.model_4/layer_normalization_34/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :p
.model_4/layer_normalization_34/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
,model_4/layer_normalization_34/Reshape/shapePack7model_4/layer_normalization_34/Reshape/shape/0:output:0&model_4/layer_normalization_34/mul:z:0(model_4/layer_normalization_34/mul_1:z:07model_4/layer_normalization_34/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
&model_4/layer_normalization_34/ReshapeReshape#model_4/dense_38/Relu:activations:05model_4/layer_normalization_34/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
*model_4/layer_normalization_34/ones/packedPack&model_4/layer_normalization_34/mul:z:0*
N*
T0*
_output_shapes
:n
)model_4/layer_normalization_34/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#model_4/layer_normalization_34/onesFill3model_4/layer_normalization_34/ones/packed:output:02model_4/layer_normalization_34/ones/Const:output:0*
T0*#
_output_shapes
:??????????
+model_4/layer_normalization_34/zeros/packedPack&model_4/layer_normalization_34/mul:z:0*
N*
T0*
_output_shapes
:o
*model_4/layer_normalization_34/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
$model_4/layer_normalization_34/zerosFill4model_4/layer_normalization_34/zeros/packed:output:03model_4/layer_normalization_34/zeros/Const:output:0*
T0*#
_output_shapes
:?????????g
$model_4/layer_normalization_34/ConstConst*
_output_shapes
: *
dtype0*
valueB i
&model_4/layer_normalization_34/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
/model_4/layer_normalization_34/FusedBatchNormV3FusedBatchNormV3/model_4/layer_normalization_34/Reshape:output:0,model_4/layer_normalization_34/ones:output:0-model_4/layer_normalization_34/zeros:output:0-model_4/layer_normalization_34/Const:output:0/model_4/layer_normalization_34/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
(model_4/layer_normalization_34/Reshape_1Reshape3model_4/layer_normalization_34/FusedBatchNormV3:y:0-model_4/layer_normalization_34/Shape:output:0*
T0*'
_output_shapes
:??????????
3model_4/layer_normalization_34/mul_2/ReadVariableOpReadVariableOp<model_4_layer_normalization_34_mul_2_readvariableop_resource*
_output_shapes
:*
dtype0?
$model_4/layer_normalization_34/mul_2Mul1model_4/layer_normalization_34/Reshape_1:output:0;model_4/layer_normalization_34/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1model_4/layer_normalization_34/add/ReadVariableOpReadVariableOp:model_4_layer_normalization_34_add_readvariableop_resource*
_output_shapes
:*
dtype0?
"model_4/layer_normalization_34/addAddV2(model_4/layer_normalization_34/mul_2:z:09model_4/layer_normalization_34/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_4/dense_39/MatMul/ReadVariableOpReadVariableOp/model_4_dense_39_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_4/dense_39/MatMulMatMul&model_4/layer_normalization_34/add:z:0.model_4/dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'model_4/dense_39/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_4/dense_39/BiasAddBiasAdd!model_4/dense_39/MatMul:product:0/model_4/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
model_4/dense_39/ReluRelu!model_4/dense_39/BiasAdd:output:0*
T0*'
_output_shapes
:?????????r
IdentityIdentity#model_4/dense_39/Relu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^model_4/dense_35/BiasAdd/ReadVariableOp*^model_4/dense_35/Tensordot/ReadVariableOp(^model_4/dense_38/BiasAdd/ReadVariableOp'^model_4/dense_38/MatMul/ReadVariableOp(^model_4/dense_39/BiasAdd/ReadVariableOp'^model_4/dense_39/MatMul/ReadVariableOp2^model_4/layer_normalization_31/add/ReadVariableOp4^model_4/layer_normalization_31/mul_3/ReadVariableOp2^model_4/layer_normalization_34/add/ReadVariableOp4^model_4/layer_normalization_34/mul_2/ReadVariableOpL^model_4/transformer_block_9/layer_normalization_32/batchnorm/ReadVariableOpP^model_4/transformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOpL^model_4/transformer_block_9/layer_normalization_33/batchnorm/ReadVariableOpP^model_4/transformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOpS^model_4/transformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOpU^model_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOpS^model_4/transformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOpU^model_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOpU^model_4/transformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOpW^model_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOpU^model_4/transformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOpW^model_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOpI^model_4/transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOpK^model_4/transformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOpI^model_4/transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOpK^model_4/transformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'model_4/dense_35/BiasAdd/ReadVariableOp'model_4/dense_35/BiasAdd/ReadVariableOp2V
)model_4/dense_35/Tensordot/ReadVariableOp)model_4/dense_35/Tensordot/ReadVariableOp2R
'model_4/dense_38/BiasAdd/ReadVariableOp'model_4/dense_38/BiasAdd/ReadVariableOp2P
&model_4/dense_38/MatMul/ReadVariableOp&model_4/dense_38/MatMul/ReadVariableOp2R
'model_4/dense_39/BiasAdd/ReadVariableOp'model_4/dense_39/BiasAdd/ReadVariableOp2P
&model_4/dense_39/MatMul/ReadVariableOp&model_4/dense_39/MatMul/ReadVariableOp2f
1model_4/layer_normalization_31/add/ReadVariableOp1model_4/layer_normalization_31/add/ReadVariableOp2j
3model_4/layer_normalization_31/mul_3/ReadVariableOp3model_4/layer_normalization_31/mul_3/ReadVariableOp2f
1model_4/layer_normalization_34/add/ReadVariableOp1model_4/layer_normalization_34/add/ReadVariableOp2j
3model_4/layer_normalization_34/mul_2/ReadVariableOp3model_4/layer_normalization_34/mul_2/ReadVariableOp2?
Kmodel_4/transformer_block_9/layer_normalization_32/batchnorm/ReadVariableOpKmodel_4/transformer_block_9/layer_normalization_32/batchnorm/ReadVariableOp2?
Omodel_4/transformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOpOmodel_4/transformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOp2?
Kmodel_4/transformer_block_9/layer_normalization_33/batchnorm/ReadVariableOpKmodel_4/transformer_block_9/layer_normalization_33/batchnorm/ReadVariableOp2?
Omodel_4/transformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOpOmodel_4/transformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOp2?
Rmodel_4/transformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOpRmodel_4/transformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOp2?
Tmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOpTmodel_4/transformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOp2?
Rmodel_4/transformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOpRmodel_4/transformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOp2?
Tmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOpTmodel_4/transformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOp2?
Tmodel_4/transformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOpTmodel_4/transformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOp2?
Vmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOpVmodel_4/transformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOp2?
Tmodel_4/transformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOpTmodel_4/transformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOp2?
Vmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOpVmodel_4/transformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOp2?
Hmodel_4/transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOpHmodel_4/transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOp2?
Jmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOpJmodel_4/transformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOp2?
Hmodel_4/transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOpHmodel_4/transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOp2?
Jmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOpJmodel_4/transformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp:U Q
+
_output_shapes
:?????????pd
"
_user_specified_name
input_10
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_255955

inputs!
dense_36_255913:
dense_36_255915:!
dense_37_255949:
dense_37_255951:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_255913dense_36_255915*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_255912?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_255949dense_37_255951*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_255948|
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????p: : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
)__inference_dense_38_layer_call_fn_259252

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_256486o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_dense_39_layer_call_and_return_conditional_losses_256557

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_39/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_39/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_39/kernel/Regularizer/Square/ReadVariableOp1dense_39/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_layer_normalization_34_layer_call_fn_259278

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_34_layer_call_and_return_conditional_losses_256534o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
D__inference_dense_35_layer_call_and_return_conditional_losses_258542

inputs3
!tensordot_readvariableop_resource:d@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?1dense_35/kernel/Regularizer/Square/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:d@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????pd?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????p@?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:d@*
dtype0?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d@r
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_35/kernel/Regularizer/SumSum&dense_35/kernel/Regularizer/Square:y:0*dense_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????p@?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp2^dense_35/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????pd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2f
1dense_35/kernel/Regularizer/Square/ReadVariableOp1dense_35/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????pd
 
_user_specified_nameinputs
?
?
)__inference_dense_35_layer_call_fn_258505

inputs
unknown:d@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_256111s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????pd: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????pd
 
_user_specified_nameinputs
?
?
D__inference_dense_36_layer_call_and_return_conditional_losses_255912

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pT
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????pe
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????pz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????p: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_256067
dense_36_input!
dense_36_256056:
dense_36_256058:!
dense_37_256061:
dense_37_256063:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_256056dense_36_256058*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_255912?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_256061dense_37_256063*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_255948|
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????p: : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:[ W
+
_output_shapes
:?????????p
(
_user_specified_namedense_36_input
??
?
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_258942

inputsU
Cmulti_head_self_attention_9_query_tensordot_readvariableop_resource:@O
Amulti_head_self_attention_9_query_biasadd_readvariableop_resource:S
Amulti_head_self_attention_9_key_tensordot_readvariableop_resource:@M
?multi_head_self_attention_9_key_biasadd_readvariableop_resource:U
Cmulti_head_self_attention_9_value_tensordot_readvariableop_resource:@O
Amulti_head_self_attention_9_value_biasadd_readvariableop_resource:S
Amulti_head_self_attention_9_out_tensordot_readvariableop_resource:M
?multi_head_self_attention_9_out_biasadd_readvariableop_resource:J
<layer_normalization_32_batchnorm_mul_readvariableop_resource:F
8layer_normalization_32_batchnorm_readvariableop_resource:I
7sequential_9_dense_36_tensordot_readvariableop_resource:C
5sequential_9_dense_36_biasadd_readvariableop_resource:I
7sequential_9_dense_37_tensordot_readvariableop_resource:C
5sequential_9_dense_37_biasadd_readvariableop_resource:J
<layer_normalization_33_batchnorm_mul_readvariableop_resource:F
8layer_normalization_33_batchnorm_readvariableop_resource:
identity??/layer_normalization_32/batchnorm/ReadVariableOp?3layer_normalization_32/batchnorm/mul/ReadVariableOp?/layer_normalization_33/batchnorm/ReadVariableOp?3layer_normalization_33/batchnorm/mul/ReadVariableOp?6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp?8multi_head_self_attention_9/key/Tensordot/ReadVariableOp?6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp?8multi_head_self_attention_9/out/Tensordot/ReadVariableOp?8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp?:multi_head_self_attention_9/query/Tensordot/ReadVariableOp?8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp?:multi_head_self_attention_9/value/Tensordot/ReadVariableOp?,sequential_9/dense_36/BiasAdd/ReadVariableOp?.sequential_9/dense_36/Tensordot/ReadVariableOp?,sequential_9/dense_37/BiasAdd/ReadVariableOp?.sequential_9/dense_37/Tensordot/ReadVariableOpW
!multi_head_self_attention_9/ShapeShapeinputs*
T0*
_output_shapes
:y
/multi_head_self_attention_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1multi_head_self_attention_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1multi_head_self_attention_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)multi_head_self_attention_9/strided_sliceStridedSlice*multi_head_self_attention_9/Shape:output:08multi_head_self_attention_9/strided_slice/stack:output:0:multi_head_self_attention_9/strided_slice/stack_1:output:0:multi_head_self_attention_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:multi_head_self_attention_9/query/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_9_query_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0z
0multi_head_self_attention_9/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0multi_head_self_attention_9/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
1multi_head_self_attention_9/query/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention_9/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/query/Tensordot/GatherV2GatherV2:multi_head_self_attention_9/query/Tensordot/Shape:output:09multi_head_self_attention_9/query/Tensordot/free:output:0Bmulti_head_self_attention_9/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;multi_head_self_attention_9/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6multi_head_self_attention_9/query/Tensordot/GatherV2_1GatherV2:multi_head_self_attention_9/query/Tensordot/Shape:output:09multi_head_self_attention_9/query/Tensordot/axes:output:0Dmulti_head_self_attention_9/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1multi_head_self_attention_9/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/query/Tensordot/ProdProd=multi_head_self_attention_9/query/Tensordot/GatherV2:output:0:multi_head_self_attention_9/query/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention_9/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2multi_head_self_attention_9/query/Tensordot/Prod_1Prod?multi_head_self_attention_9/query/Tensordot/GatherV2_1:output:0<multi_head_self_attention_9/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention_9/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/query/Tensordot/concatConcatV29multi_head_self_attention_9/query/Tensordot/free:output:09multi_head_self_attention_9/query/Tensordot/axes:output:0@multi_head_self_attention_9/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1multi_head_self_attention_9/query/Tensordot/stackPack9multi_head_self_attention_9/query/Tensordot/Prod:output:0;multi_head_self_attention_9/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5multi_head_self_attention_9/query/Tensordot/transpose	Transposeinputs;multi_head_self_attention_9/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
3multi_head_self_attention_9/query/Tensordot/ReshapeReshape9multi_head_self_attention_9/query/Tensordot/transpose:y:0:multi_head_self_attention_9/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2multi_head_self_attention_9/query/Tensordot/MatMulMatMul<multi_head_self_attention_9/query/Tensordot/Reshape:output:0Bmulti_head_self_attention_9/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
3multi_head_self_attention_9/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:{
9multi_head_self_attention_9/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/query/Tensordot/concat_1ConcatV2=multi_head_self_attention_9/query/Tensordot/GatherV2:output:0<multi_head_self_attention_9/query/Tensordot/Const_2:output:0Bmulti_head_self_attention_9/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+multi_head_self_attention_9/query/TensordotReshape<multi_head_self_attention_9/query/Tensordot/MatMul:product:0=multi_head_self_attention_9/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/query/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_query_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_self_attention_9/query/BiasAddBiasAdd4multi_head_self_attention_9/query/Tensordot:output:0@multi_head_self_attention_9/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/key/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_key_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0x
.multi_head_self_attention_9/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention_9/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_self_attention_9/key/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_self_attention_9/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/key/Tensordot/GatherV2GatherV28multi_head_self_attention_9/key/Tensordot/Shape:output:07multi_head_self_attention_9/key/Tensordot/free:output:0@multi_head_self_attention_9/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention_9/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/key/Tensordot/GatherV2_1GatherV28multi_head_self_attention_9/key/Tensordot/Shape:output:07multi_head_self_attention_9/key/Tensordot/axes:output:0Bmulti_head_self_attention_9/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention_9/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_self_attention_9/key/Tensordot/ProdProd;multi_head_self_attention_9/key/Tensordot/GatherV2:output:08multi_head_self_attention_9/key/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention_9/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/key/Tensordot/Prod_1Prod=multi_head_self_attention_9/key/Tensordot/GatherV2_1:output:0:multi_head_self_attention_9/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention_9/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_self_attention_9/key/Tensordot/concatConcatV27multi_head_self_attention_9/key/Tensordot/free:output:07multi_head_self_attention_9/key/Tensordot/axes:output:0>multi_head_self_attention_9/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_self_attention_9/key/Tensordot/stackPack7multi_head_self_attention_9/key/Tensordot/Prod:output:09multi_head_self_attention_9/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_self_attention_9/key/Tensordot/transpose	Transposeinputs9multi_head_self_attention_9/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
1multi_head_self_attention_9/key/Tensordot/ReshapeReshape7multi_head_self_attention_9/key/Tensordot/transpose:y:08multi_head_self_attention_9/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_self_attention_9/key/Tensordot/MatMulMatMul:multi_head_self_attention_9/key/Tensordot/Reshape:output:0@multi_head_self_attention_9/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_self_attention_9/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_self_attention_9/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/key/Tensordot/concat_1ConcatV2;multi_head_self_attention_9/key/Tensordot/GatherV2:output:0:multi_head_self_attention_9/key/Tensordot/Const_2:output:0@multi_head_self_attention_9/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_self_attention_9/key/TensordotReshape:multi_head_self_attention_9/key/Tensordot/MatMul:product:0;multi_head_self_attention_9/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
6multi_head_self_attention_9/key/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_9_key_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_self_attention_9/key/BiasAddBiasAdd2multi_head_self_attention_9/key/Tensordot:output:0>multi_head_self_attention_9/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
:multi_head_self_attention_9/value/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_9_value_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0z
0multi_head_self_attention_9/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0multi_head_self_attention_9/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
1multi_head_self_attention_9/value/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention_9/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/value/Tensordot/GatherV2GatherV2:multi_head_self_attention_9/value/Tensordot/Shape:output:09multi_head_self_attention_9/value/Tensordot/free:output:0Bmulti_head_self_attention_9/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;multi_head_self_attention_9/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6multi_head_self_attention_9/value/Tensordot/GatherV2_1GatherV2:multi_head_self_attention_9/value/Tensordot/Shape:output:09multi_head_self_attention_9/value/Tensordot/axes:output:0Dmulti_head_self_attention_9/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1multi_head_self_attention_9/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/value/Tensordot/ProdProd=multi_head_self_attention_9/value/Tensordot/GatherV2:output:0:multi_head_self_attention_9/value/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention_9/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2multi_head_self_attention_9/value/Tensordot/Prod_1Prod?multi_head_self_attention_9/value/Tensordot/GatherV2_1:output:0<multi_head_self_attention_9/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention_9/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/value/Tensordot/concatConcatV29multi_head_self_attention_9/value/Tensordot/free:output:09multi_head_self_attention_9/value/Tensordot/axes:output:0@multi_head_self_attention_9/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1multi_head_self_attention_9/value/Tensordot/stackPack9multi_head_self_attention_9/value/Tensordot/Prod:output:0;multi_head_self_attention_9/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5multi_head_self_attention_9/value/Tensordot/transpose	Transposeinputs;multi_head_self_attention_9/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
3multi_head_self_attention_9/value/Tensordot/ReshapeReshape9multi_head_self_attention_9/value/Tensordot/transpose:y:0:multi_head_self_attention_9/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2multi_head_self_attention_9/value/Tensordot/MatMulMatMul<multi_head_self_attention_9/value/Tensordot/Reshape:output:0Bmulti_head_self_attention_9/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
3multi_head_self_attention_9/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:{
9multi_head_self_attention_9/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/value/Tensordot/concat_1ConcatV2=multi_head_self_attention_9/value/Tensordot/GatherV2:output:0<multi_head_self_attention_9/value/Tensordot/Const_2:output:0Bmulti_head_self_attention_9/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+multi_head_self_attention_9/value/TensordotReshape<multi_head_self_attention_9/value/Tensordot/MatMul:product:0=multi_head_self_attention_9/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/value/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_self_attention_9/value/BiasAddBiasAdd4multi_head_self_attention_9/value/Tensordot:output:0@multi_head_self_attention_9/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pm
+multi_head_self_attention_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :pm
+multi_head_self_attention_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention_9/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
)multi_head_self_attention_9/Reshape/shapePack2multi_head_self_attention_9/strided_slice:output:04multi_head_self_attention_9/Reshape/shape/1:output:04multi_head_self_attention_9/Reshape/shape/2:output:04multi_head_self_attention_9/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
#multi_head_self_attention_9/ReshapeReshape2multi_head_self_attention_9/query/BiasAdd:output:02multi_head_self_attention_9/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????p?
*multi_head_self_attention_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
%multi_head_self_attention_9/transpose	Transpose,multi_head_self_attention_9/Reshape:output:03multi_head_self_attention_9/transpose/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-multi_head_self_attention_9/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_1/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_1/shape/1:output:06multi_head_self_attention_9/Reshape_1/shape/2:output:06multi_head_self_attention_9/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_1Reshape0multi_head_self_attention_9/key/BiasAdd:output:04multi_head_self_attention_9/Reshape_1/shape:output:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_1	Transpose.multi_head_self_attention_9/Reshape_1:output:05multi_head_self_attention_9/transpose_1/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-multi_head_self_attention_9/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_2/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_2/shape/1:output:06multi_head_self_attention_9/Reshape_2/shape/2:output:06multi_head_self_attention_9/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_2Reshape2multi_head_self_attention_9/value/BiasAdd:output:04multi_head_self_attention_9/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_2	Transpose.multi_head_self_attention_9/Reshape_2:output:05multi_head_self_attention_9/transpose_2/perm:output:0*
T0*/
_output_shapes
:?????????p?
"multi_head_self_attention_9/MatMulBatchMatMulV2)multi_head_self_attention_9/transpose:y:0+multi_head_self_attention_9/transpose_1:y:0*
T0*/
_output_shapes
:?????????pp*
adj_y(~
#multi_head_self_attention_9/Shape_1Shape+multi_head_self_attention_9/transpose_1:y:0*
T0*
_output_shapes
:?
1multi_head_self_attention_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
3multi_head_self_attention_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3multi_head_self_attention_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+multi_head_self_attention_9/strided_slice_1StridedSlice,multi_head_self_attention_9/Shape_1:output:0:multi_head_self_attention_9/strided_slice_1/stack:output:0<multi_head_self_attention_9/strided_slice_1/stack_1:output:0<multi_head_self_attention_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
 multi_head_self_attention_9/CastCast4multi_head_self_attention_9/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: o
 multi_head_self_attention_9/SqrtSqrt$multi_head_self_attention_9/Cast:y:0*
T0*
_output_shapes
: ?
#multi_head_self_attention_9/truedivRealDiv+multi_head_self_attention_9/MatMul:output:0$multi_head_self_attention_9/Sqrt:y:0*
T0*/
_output_shapes
:?????????pp?
#multi_head_self_attention_9/SoftmaxSoftmax'multi_head_self_attention_9/truediv:z:0*
T0*/
_output_shapes
:?????????pp?
$multi_head_self_attention_9/MatMul_1BatchMatMulV2-multi_head_self_attention_9/Softmax:softmax:0+multi_head_self_attention_9/transpose_2:y:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_3	Transpose-multi_head_self_attention_9/MatMul_1:output:05multi_head_self_attention_9/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_3/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_3/shape/1:output:06multi_head_self_attention_9/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_3Reshape+multi_head_self_attention_9/transpose_3:y:04multi_head_self_attention_9/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/out/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_self_attention_9/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention_9/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
/multi_head_self_attention_9/out/Tensordot/ShapeShape.multi_head_self_attention_9/Reshape_3:output:0*
T0*
_output_shapes
:y
7multi_head_self_attention_9/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/out/Tensordot/GatherV2GatherV28multi_head_self_attention_9/out/Tensordot/Shape:output:07multi_head_self_attention_9/out/Tensordot/free:output:0@multi_head_self_attention_9/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention_9/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/out/Tensordot/GatherV2_1GatherV28multi_head_self_attention_9/out/Tensordot/Shape:output:07multi_head_self_attention_9/out/Tensordot/axes:output:0Bmulti_head_self_attention_9/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention_9/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_self_attention_9/out/Tensordot/ProdProd;multi_head_self_attention_9/out/Tensordot/GatherV2:output:08multi_head_self_attention_9/out/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention_9/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/out/Tensordot/Prod_1Prod=multi_head_self_attention_9/out/Tensordot/GatherV2_1:output:0:multi_head_self_attention_9/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention_9/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_self_attention_9/out/Tensordot/concatConcatV27multi_head_self_attention_9/out/Tensordot/free:output:07multi_head_self_attention_9/out/Tensordot/axes:output:0>multi_head_self_attention_9/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_self_attention_9/out/Tensordot/stackPack7multi_head_self_attention_9/out/Tensordot/Prod:output:09multi_head_self_attention_9/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_self_attention_9/out/Tensordot/transpose	Transpose.multi_head_self_attention_9/Reshape_3:output:09multi_head_self_attention_9/out/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
1multi_head_self_attention_9/out/Tensordot/ReshapeReshape7multi_head_self_attention_9/out/Tensordot/transpose:y:08multi_head_self_attention_9/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_self_attention_9/out/Tensordot/MatMulMatMul:multi_head_self_attention_9/out/Tensordot/Reshape:output:0@multi_head_self_attention_9/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_self_attention_9/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_self_attention_9/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/out/Tensordot/concat_1ConcatV2;multi_head_self_attention_9/out/Tensordot/GatherV2:output:0:multi_head_self_attention_9/out/Tensordot/Const_2:output:0@multi_head_self_attention_9/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_self_attention_9/out/TensordotReshape:multi_head_self_attention_9/out/Tensordot/MatMul:product:0;multi_head_self_attention_9/out/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
6multi_head_self_attention_9/out/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_9_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_self_attention_9/out/BiasAddBiasAdd2multi_head_self_attention_9/out/Tensordot:output:0>multi_head_self_attention_9/out/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
dropout_28/IdentityIdentity0multi_head_self_attention_9/out/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p
5layer_normalization_32/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_32/moments/meanMeandropout_28/Identity:output:0>layer_normalization_32/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
+layer_normalization_32/moments/StopGradientStopGradient,layer_normalization_32/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
0layer_normalization_32/moments/SquaredDifferenceSquaredDifferencedropout_28/Identity:output:04layer_normalization_32/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
9layer_normalization_32/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
'layer_normalization_32/moments/varianceMean4layer_normalization_32/moments/SquaredDifference:z:0Blayer_normalization_32/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(k
&layer_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
$layer_normalization_32/batchnorm/addAddV20layer_normalization_32/moments/variance:output:0/layer_normalization_32/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/RsqrtRsqrt(layer_normalization_32/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
3layer_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_32/batchnorm/mulMul*layer_normalization_32/batchnorm/Rsqrt:y:0;layer_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/mul_1Muldropout_28/Identity:output:0(layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/mul_2Mul,layer_normalization_32/moments/mean:output:0(layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
/layer_normalization_32/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_32/batchnorm/subSub7layer_normalization_32/batchnorm/ReadVariableOp:value:0*layer_normalization_32/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/add_1AddV2*layer_normalization_32/batchnorm/mul_1:z:0(layer_normalization_32/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p?
.sequential_9/dense_36/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_36_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_9/dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_9/dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_9/dense_36/Tensordot/ShapeShape*layer_normalization_32/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_9/dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_36/Tensordot/GatherV2GatherV2.sequential_9/dense_36/Tensordot/Shape:output:0-sequential_9/dense_36/Tensordot/free:output:06sequential_9/dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_9/dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_9/dense_36/Tensordot/GatherV2_1GatherV2.sequential_9/dense_36/Tensordot/Shape:output:0-sequential_9/dense_36/Tensordot/axes:output:08sequential_9/dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_9/dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_9/dense_36/Tensordot/ProdProd1sequential_9/dense_36/Tensordot/GatherV2:output:0.sequential_9/dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_9/dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_9/dense_36/Tensordot/Prod_1Prod3sequential_9/dense_36/Tensordot/GatherV2_1:output:00sequential_9/dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_9/dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_9/dense_36/Tensordot/concatConcatV2-sequential_9/dense_36/Tensordot/free:output:0-sequential_9/dense_36/Tensordot/axes:output:04sequential_9/dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_9/dense_36/Tensordot/stackPack-sequential_9/dense_36/Tensordot/Prod:output:0/sequential_9/dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_9/dense_36/Tensordot/transpose	Transpose*layer_normalization_32/batchnorm/add_1:z:0/sequential_9/dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
'sequential_9/dense_36/Tensordot/ReshapeReshape-sequential_9/dense_36/Tensordot/transpose:y:0.sequential_9/dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_9/dense_36/Tensordot/MatMulMatMul0sequential_9/dense_36/Tensordot/Reshape:output:06sequential_9/dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_9/dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_9/dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_36/Tensordot/concat_1ConcatV21sequential_9/dense_36/Tensordot/GatherV2:output:00sequential_9/dense_36/Tensordot/Const_2:output:06sequential_9/dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_9/dense_36/TensordotReshape0sequential_9/dense_36/Tensordot/MatMul:product:01sequential_9/dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/dense_36/BiasAddBiasAdd(sequential_9/dense_36/Tensordot:output:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
.sequential_9/dense_37/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_37_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_9/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_9/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_9/dense_37/Tensordot/ShapeShape(sequential_9/dense_36/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_9/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_37/Tensordot/GatherV2GatherV2.sequential_9/dense_37/Tensordot/Shape:output:0-sequential_9/dense_37/Tensordot/free:output:06sequential_9/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_9/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_9/dense_37/Tensordot/GatherV2_1GatherV2.sequential_9/dense_37/Tensordot/Shape:output:0-sequential_9/dense_37/Tensordot/axes:output:08sequential_9/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_9/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_9/dense_37/Tensordot/ProdProd1sequential_9/dense_37/Tensordot/GatherV2:output:0.sequential_9/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_9/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_9/dense_37/Tensordot/Prod_1Prod3sequential_9/dense_37/Tensordot/GatherV2_1:output:00sequential_9/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_9/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_9/dense_37/Tensordot/concatConcatV2-sequential_9/dense_37/Tensordot/free:output:0-sequential_9/dense_37/Tensordot/axes:output:04sequential_9/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_9/dense_37/Tensordot/stackPack-sequential_9/dense_37/Tensordot/Prod:output:0/sequential_9/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_9/dense_37/Tensordot/transpose	Transpose(sequential_9/dense_36/Relu:activations:0/sequential_9/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
'sequential_9/dense_37/Tensordot/ReshapeReshape-sequential_9/dense_37/Tensordot/transpose:y:0.sequential_9/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_9/dense_37/Tensordot/MatMulMatMul0sequential_9/dense_37/Tensordot/Reshape:output:06sequential_9/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_9/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_9/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_37/Tensordot/concat_1ConcatV21sequential_9/dense_37/Tensordot/GatherV2:output:00sequential_9/dense_37/Tensordot/Const_2:output:06sequential_9/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_9/dense_37/TensordotReshape0sequential_9/dense_37/Tensordot/MatMul:product:01sequential_9/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/dense_37/BiasAddBiasAdd(sequential_9/dense_37/Tensordot:output:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p}
dropout_29/IdentityIdentity&sequential_9/dense_37/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
addAddV2*layer_normalization_32/batchnorm/add_1:z:0dropout_29/Identity:output:0*
T0*+
_output_shapes
:?????????p
5layer_normalization_33/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_33/moments/meanMeanadd:z:0>layer_normalization_33/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
+layer_normalization_33/moments/StopGradientStopGradient,layer_normalization_33/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
0layer_normalization_33/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_33/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
9layer_normalization_33/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
'layer_normalization_33/moments/varianceMean4layer_normalization_33/moments/SquaredDifference:z:0Blayer_normalization_33/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(k
&layer_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
$layer_normalization_33/batchnorm/addAddV20layer_normalization_33/moments/variance:output:0/layer_normalization_33/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/RsqrtRsqrt(layer_normalization_33/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
3layer_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_33/batchnorm/mulMul*layer_normalization_33/batchnorm/Rsqrt:y:0;layer_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/mul_1Muladd:z:0(layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/mul_2Mul,layer_normalization_33/moments/mean:output:0(layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
/layer_normalization_33/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_33/batchnorm/subSub7layer_normalization_33/batchnorm/ReadVariableOp:value:0*layer_normalization_33/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/add_1AddV2*layer_normalization_33/batchnorm/mul_1:z:0(layer_normalization_33/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p}
IdentityIdentity*layer_normalization_33/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp0^layer_normalization_32/batchnorm/ReadVariableOp4^layer_normalization_32/batchnorm/mul/ReadVariableOp0^layer_normalization_33/batchnorm/ReadVariableOp4^layer_normalization_33/batchnorm/mul/ReadVariableOp7^multi_head_self_attention_9/key/BiasAdd/ReadVariableOp9^multi_head_self_attention_9/key/Tensordot/ReadVariableOp7^multi_head_self_attention_9/out/BiasAdd/ReadVariableOp9^multi_head_self_attention_9/out/Tensordot/ReadVariableOp9^multi_head_self_attention_9/query/BiasAdd/ReadVariableOp;^multi_head_self_attention_9/query/Tensordot/ReadVariableOp9^multi_head_self_attention_9/value/BiasAdd/ReadVariableOp;^multi_head_self_attention_9/value/Tensordot/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp/^sequential_9/dense_36/Tensordot/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp/^sequential_9/dense_37/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????p@: : : : : : : : : : : : : : : : 2b
/layer_normalization_32/batchnorm/ReadVariableOp/layer_normalization_32/batchnorm/ReadVariableOp2j
3layer_normalization_32/batchnorm/mul/ReadVariableOp3layer_normalization_32/batchnorm/mul/ReadVariableOp2b
/layer_normalization_33/batchnorm/ReadVariableOp/layer_normalization_33/batchnorm/ReadVariableOp2j
3layer_normalization_33/batchnorm/mul/ReadVariableOp3layer_normalization_33/batchnorm/mul/ReadVariableOp2p
6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp2t
8multi_head_self_attention_9/key/Tensordot/ReadVariableOp8multi_head_self_attention_9/key/Tensordot/ReadVariableOp2p
6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp2t
8multi_head_self_attention_9/out/Tensordot/ReadVariableOp8multi_head_self_attention_9/out/Tensordot/ReadVariableOp2t
8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp2x
:multi_head_self_attention_9/query/Tensordot/ReadVariableOp:multi_head_self_attention_9/query/Tensordot/ReadVariableOp2t
8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp2x
:multi_head_self_attention_9/value/Tensordot/ReadVariableOp:multi_head_self_attention_9/value/Tensordot/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2`
.sequential_9/dense_36/Tensordot/ReadVariableOp.sequential_9/dense_36/Tensordot/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2`
.sequential_9/dense_37/Tensordot/ReadVariableOp.sequential_9/dense_37/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?

e
F__inference_dropout_27_layer_call_and_return_conditional_losses_258569

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????p@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????p@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????p@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????p@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????p@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????p@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????p@:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_259374M
:dense_38_kernel_regularizer_square_readvariableop_resource:	?
identity??1dense_38/kernel/Regularizer/Square/ReadVariableOp?
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_38_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	?*
dtype0?
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?r
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_38/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp
?
d
F__inference_dropout_30_layer_call_and_return_conditional_losses_256467

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?L
?
C__inference_model_4_layer_call_and_return_conditional_losses_257408
input_10!
dense_35_257328:d@
dense_35_257330:@+
layer_normalization_31_257334:@+
layer_normalization_31_257336:@,
transformer_block_9_257339:@(
transformer_block_9_257341:,
transformer_block_9_257343:@(
transformer_block_9_257345:,
transformer_block_9_257347:@(
transformer_block_9_257349:,
transformer_block_9_257351:(
transformer_block_9_257353:(
transformer_block_9_257355:(
transformer_block_9_257357:,
transformer_block_9_257359:(
transformer_block_9_257361:,
transformer_block_9_257363:(
transformer_block_9_257365:(
transformer_block_9_257367:(
transformer_block_9_257369:"
dense_38_257374:	?
dense_38_257376:+
layer_normalization_34_257379:+
layer_normalization_34_257381:!
dense_39_257384:
dense_39_257386:
identity?? dense_35/StatefulPartitionedCall?1dense_35/kernel/Regularizer/Square/ReadVariableOp? dense_38/StatefulPartitionedCall?1dense_38/kernel/Regularizer/Square/ReadVariableOp? dense_39/StatefulPartitionedCall?1dense_39/kernel/Regularizer/Square/ReadVariableOp?.layer_normalization_31/StatefulPartitionedCall?.layer_normalization_34/StatefulPartitionedCall?+transformer_block_9/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCallinput_10dense_35_257328dense_35_257330*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_256111?
dropout_27/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_256122?
.layer_normalization_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0layer_normalization_31_257334layer_normalization_31_257336*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_31_layer_call_and_return_conditional_losses_256171?
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_31/StatefulPartitionedCall:output:0transformer_block_9_257339transformer_block_9_257341transformer_block_9_257343transformer_block_9_257345transformer_block_9_257347transformer_block_9_257349transformer_block_9_257351transformer_block_9_257353transformer_block_9_257355transformer_block_9_257357transformer_block_9_257359transformer_block_9_257361transformer_block_9_257363transformer_block_9_257365transformer_block_9_257367transformer_block_9_257369*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_256420?
flatten_4/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_256460?
dropout_30/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_256467?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0dense_38_257374dense_38_257376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_256486?
.layer_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0layer_normalization_34_257379layer_normalization_34_257381*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_34_layer_call_and_return_conditional_losses_256534?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_34/StatefulPartitionedCall:output:0dense_39_257384dense_39_257386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_256557?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_257328*
_output_shapes

:d@*
dtype0?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d@r
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_35/kernel/Regularizer/SumSum&dense_35/kernel/Regularizer/Square:y:0*dense_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38_257374*
_output_shapes
:	?*
dtype0?
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?r
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_39_257384*
_output_shapes

:*
dtype0?
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_35/StatefulPartitionedCall2^dense_35/kernel/Regularizer/Square/ReadVariableOp!^dense_38/StatefulPartitionedCall2^dense_38/kernel/Regularizer/Square/ReadVariableOp!^dense_39/StatefulPartitionedCall2^dense_39/kernel/Regularizer/Square/ReadVariableOp/^layer_normalization_31/StatefulPartitionedCall/^layer_normalization_34/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2f
1dense_35/kernel/Regularizer/Square/ReadVariableOp1dense_35/kernel/Regularizer/Square/ReadVariableOp2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2f
1dense_39/kernel/Regularizer/Square/ReadVariableOp1dense_39/kernel/Regularizer/Square/ReadVariableOp2`
.layer_normalization_31/StatefulPartitionedCall.layer_normalization_31/StatefulPartitionedCall2`
.layer_normalization_34/StatefulPartitionedCall.layer_normalization_34/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????pd
"
_user_specified_name
input_10
?
d
+__inference_dropout_30_layer_call_fn_259220

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_256687p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_dense_37_layer_call_and_return_conditional_losses_255948

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????pz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????p: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_259525

inputs<
*dense_36_tensordot_readvariableop_resource:6
(dense_36_biasadd_readvariableop_resource:<
*dense_37_tensordot_readvariableop_resource:6
(dense_37_biasadd_readvariableop_resource:
identity??dense_36/BiasAdd/ReadVariableOp?!dense_36/Tensordot/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?!dense_37/Tensordot/ReadVariableOp?
!dense_36/Tensordot/ReadVariableOpReadVariableOp*dense_36_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_36/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_36/Tensordot/GatherV2GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/free:output:0)dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_36/Tensordot/GatherV2_1GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/axes:output:0+dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_36/Tensordot/ProdProd$dense_36/Tensordot/GatherV2:output:0!dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_36/Tensordot/Prod_1Prod&dense_36/Tensordot/GatherV2_1:output:0#dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_36/Tensordot/concatConcatV2 dense_36/Tensordot/free:output:0 dense_36/Tensordot/axes:output:0'dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_36/Tensordot/stackPack dense_36/Tensordot/Prod:output:0"dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_36/Tensordot/transpose	Transposeinputs"dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
dense_36/Tensordot/ReshapeReshape dense_36/Tensordot/transpose:y:0!dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_36/Tensordot/MatMulMatMul#dense_36/Tensordot/Reshape:output:0)dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_36/Tensordot/concat_1ConcatV2$dense_36/Tensordot/GatherV2:output:0#dense_36/Tensordot/Const_2:output:0)dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_36/TensordotReshape#dense_36/Tensordot/MatMul:product:0$dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_36/BiasAddBiasAdddense_36/Tensordot:output:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pf
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
!dense_37/Tensordot/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_37/Tensordot/ShapeShapedense_36/Relu:activations:0*
T0*
_output_shapes
:b
 dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_37/Tensordot/GatherV2GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/free:output:0)dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_37/Tensordot/GatherV2_1GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/axes:output:0+dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_37/Tensordot/ProdProd$dense_37/Tensordot/GatherV2:output:0!dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_37/Tensordot/Prod_1Prod&dense_37/Tensordot/GatherV2_1:output:0#dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_37/Tensordot/concatConcatV2 dense_37/Tensordot/free:output:0 dense_37/Tensordot/axes:output:0'dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_37/Tensordot/stackPack dense_37/Tensordot/Prod:output:0"dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_37/Tensordot/transpose	Transposedense_36/Relu:activations:0"dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
dense_37/Tensordot/ReshapeReshape dense_37/Tensordot/transpose:y:0!dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_37/Tensordot/MatMulMatMul#dense_37/Tensordot/Reshape:output:0)dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_37/Tensordot/concat_1ConcatV2$dense_37/Tensordot/GatherV2:output:0#dense_37/Tensordot/Const_2:output:0)dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_37/TensordotReshape#dense_37/Tensordot/MatMul:product:0$dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_37/BiasAddBiasAdddense_37/Tensordot:output:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pl
IdentityIdentitydense_37/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp ^dense_36/BiasAdd/ReadVariableOp"^dense_36/Tensordot/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp"^dense_37/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????p: : : : 2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2F
!dense_36/Tensordot/ReadVariableOp!dense_36/Tensordot/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2F
!dense_37/Tensordot/ReadVariableOp!dense_37/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_259398

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_255955s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????p: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
)__inference_dense_36_layer_call_fn_259534

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_255912s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????p: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?$
?
R__inference_layer_normalization_31_layer_call_and_return_conditional_losses_258625

inputs+
mul_3_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:?????????p@n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:@*
dtype0t
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0i
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????p@r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????p@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
?
D__inference_dense_37_layer_call_and_return_conditional_losses_259604

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????pz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????p: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_259411

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_256015s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????p: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
??
?A
"__inference__traced_restore_260159
file_prefix2
 assignvariableop_dense_35_kernel:d@.
 assignvariableop_1_dense_35_bias:@=
/assignvariableop_2_layer_normalization_31_gamma:@<
.assignvariableop_3_layer_normalization_31_beta:@5
"assignvariableop_4_dense_38_kernel:	?.
 assignvariableop_5_dense_38_bias:=
/assignvariableop_6_layer_normalization_34_gamma:<
.assignvariableop_7_layer_normalization_34_beta:4
"assignvariableop_8_dense_39_kernel:.
 assignvariableop_9_dense_39_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: b
Passignvariableop_15_transformer_block_9_multi_head_self_attention_9_query_kernel:@\
Nassignvariableop_16_transformer_block_9_multi_head_self_attention_9_query_bias:`
Nassignvariableop_17_transformer_block_9_multi_head_self_attention_9_key_kernel:@Z
Lassignvariableop_18_transformer_block_9_multi_head_self_attention_9_key_bias:b
Passignvariableop_19_transformer_block_9_multi_head_self_attention_9_value_kernel:@\
Nassignvariableop_20_transformer_block_9_multi_head_self_attention_9_value_bias:`
Nassignvariableop_21_transformer_block_9_multi_head_self_attention_9_out_kernel:Z
Lassignvariableop_22_transformer_block_9_multi_head_self_attention_9_out_bias:5
#assignvariableop_23_dense_36_kernel:/
!assignvariableop_24_dense_36_bias:5
#assignvariableop_25_dense_37_kernel:/
!assignvariableop_26_dense_37_bias:R
Dassignvariableop_27_transformer_block_9_layer_normalization_32_gamma:Q
Cassignvariableop_28_transformer_block_9_layer_normalization_32_beta:R
Dassignvariableop_29_transformer_block_9_layer_normalization_33_gamma:Q
Cassignvariableop_30_transformer_block_9_layer_normalization_33_beta:#
assignvariableop_31_total: #
assignvariableop_32_count: %
assignvariableop_33_total_1: %
assignvariableop_34_count_1: <
*assignvariableop_35_adam_dense_35_kernel_m:d@6
(assignvariableop_36_adam_dense_35_bias_m:@E
7assignvariableop_37_adam_layer_normalization_31_gamma_m:@D
6assignvariableop_38_adam_layer_normalization_31_beta_m:@=
*assignvariableop_39_adam_dense_38_kernel_m:	?6
(assignvariableop_40_adam_dense_38_bias_m:E
7assignvariableop_41_adam_layer_normalization_34_gamma_m:D
6assignvariableop_42_adam_layer_normalization_34_beta_m:<
*assignvariableop_43_adam_dense_39_kernel_m:6
(assignvariableop_44_adam_dense_39_bias_m:i
Wassignvariableop_45_adam_transformer_block_9_multi_head_self_attention_9_query_kernel_m:@c
Uassignvariableop_46_adam_transformer_block_9_multi_head_self_attention_9_query_bias_m:g
Uassignvariableop_47_adam_transformer_block_9_multi_head_self_attention_9_key_kernel_m:@a
Sassignvariableop_48_adam_transformer_block_9_multi_head_self_attention_9_key_bias_m:i
Wassignvariableop_49_adam_transformer_block_9_multi_head_self_attention_9_value_kernel_m:@c
Uassignvariableop_50_adam_transformer_block_9_multi_head_self_attention_9_value_bias_m:g
Uassignvariableop_51_adam_transformer_block_9_multi_head_self_attention_9_out_kernel_m:a
Sassignvariableop_52_adam_transformer_block_9_multi_head_self_attention_9_out_bias_m:<
*assignvariableop_53_adam_dense_36_kernel_m:6
(assignvariableop_54_adam_dense_36_bias_m:<
*assignvariableop_55_adam_dense_37_kernel_m:6
(assignvariableop_56_adam_dense_37_bias_m:Y
Kassignvariableop_57_adam_transformer_block_9_layer_normalization_32_gamma_m:X
Jassignvariableop_58_adam_transformer_block_9_layer_normalization_32_beta_m:Y
Kassignvariableop_59_adam_transformer_block_9_layer_normalization_33_gamma_m:X
Jassignvariableop_60_adam_transformer_block_9_layer_normalization_33_beta_m:<
*assignvariableop_61_adam_dense_35_kernel_v:d@6
(assignvariableop_62_adam_dense_35_bias_v:@E
7assignvariableop_63_adam_layer_normalization_31_gamma_v:@D
6assignvariableop_64_adam_layer_normalization_31_beta_v:@=
*assignvariableop_65_adam_dense_38_kernel_v:	?6
(assignvariableop_66_adam_dense_38_bias_v:E
7assignvariableop_67_adam_layer_normalization_34_gamma_v:D
6assignvariableop_68_adam_layer_normalization_34_beta_v:<
*assignvariableop_69_adam_dense_39_kernel_v:6
(assignvariableop_70_adam_dense_39_bias_v:i
Wassignvariableop_71_adam_transformer_block_9_multi_head_self_attention_9_query_kernel_v:@c
Uassignvariableop_72_adam_transformer_block_9_multi_head_self_attention_9_query_bias_v:g
Uassignvariableop_73_adam_transformer_block_9_multi_head_self_attention_9_key_kernel_v:@a
Sassignvariableop_74_adam_transformer_block_9_multi_head_self_attention_9_key_bias_v:i
Wassignvariableop_75_adam_transformer_block_9_multi_head_self_attention_9_value_kernel_v:@c
Uassignvariableop_76_adam_transformer_block_9_multi_head_self_attention_9_value_bias_v:g
Uassignvariableop_77_adam_transformer_block_9_multi_head_self_attention_9_out_kernel_v:a
Sassignvariableop_78_adam_transformer_block_9_multi_head_self_attention_9_out_bias_v:<
*assignvariableop_79_adam_dense_36_kernel_v:6
(assignvariableop_80_adam_dense_36_bias_v:<
*assignvariableop_81_adam_dense_37_kernel_v:6
(assignvariableop_82_adam_dense_37_bias_v:Y
Kassignvariableop_83_adam_transformer_block_9_layer_normalization_32_gamma_v:X
Jassignvariableop_84_adam_transformer_block_9_layer_normalization_32_beta_v:Y
Kassignvariableop_85_adam_transformer_block_9_layer_normalization_33_gamma_v:X
Jassignvariableop_86_adam_transformer_block_9_layer_normalization_33_beta_v:
identity_88??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_9?,
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?+
value?+B?+XB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?
value?B?XB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*f
dtypes\
Z2X	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_dense_35_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_35_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_layer_normalization_31_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_layer_normalization_31_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_38_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_38_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp/assignvariableop_6_layer_normalization_34_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_layer_normalization_34_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_39_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_39_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpPassignvariableop_15_transformer_block_9_multi_head_self_attention_9_query_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpNassignvariableop_16_transformer_block_9_multi_head_self_attention_9_query_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpNassignvariableop_17_transformer_block_9_multi_head_self_attention_9_key_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpLassignvariableop_18_transformer_block_9_multi_head_self_attention_9_key_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpPassignvariableop_19_transformer_block_9_multi_head_self_attention_9_value_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpNassignvariableop_20_transformer_block_9_multi_head_self_attention_9_value_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpNassignvariableop_21_transformer_block_9_multi_head_self_attention_9_out_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpLassignvariableop_22_transformer_block_9_multi_head_self_attention_9_out_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp#assignvariableop_23_dense_36_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp!assignvariableop_24_dense_36_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp#assignvariableop_25_dense_37_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp!assignvariableop_26_dense_37_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpDassignvariableop_27_transformer_block_9_layer_normalization_32_gammaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpCassignvariableop_28_transformer_block_9_layer_normalization_32_betaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpDassignvariableop_29_transformer_block_9_layer_normalization_33_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpCassignvariableop_30_transformer_block_9_layer_normalization_33_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_35_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_35_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_layer_normalization_31_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_layer_normalization_31_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_38_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_38_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_layer_normalization_34_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_layer_normalization_34_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_39_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_39_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpWassignvariableop_45_adam_transformer_block_9_multi_head_self_attention_9_query_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOpUassignvariableop_46_adam_transformer_block_9_multi_head_self_attention_9_query_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpUassignvariableop_47_adam_transformer_block_9_multi_head_self_attention_9_key_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOpSassignvariableop_48_adam_transformer_block_9_multi_head_self_attention_9_key_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOpWassignvariableop_49_adam_transformer_block_9_multi_head_self_attention_9_value_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOpUassignvariableop_50_adam_transformer_block_9_multi_head_self_attention_9_value_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpUassignvariableop_51_adam_transformer_block_9_multi_head_self_attention_9_out_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOpSassignvariableop_52_adam_transformer_block_9_multi_head_self_attention_9_out_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_36_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_36_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_37_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_37_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpKassignvariableop_57_adam_transformer_block_9_layer_normalization_32_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOpJassignvariableop_58_adam_transformer_block_9_layer_normalization_32_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOpKassignvariableop_59_adam_transformer_block_9_layer_normalization_33_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOpJassignvariableop_60_adam_transformer_block_9_layer_normalization_33_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_35_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_35_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_layer_normalization_31_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_layer_normalization_31_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_38_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_38_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_layer_normalization_34_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_layer_normalization_34_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_dense_39_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_dense_39_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOpWassignvariableop_71_adam_transformer_block_9_multi_head_self_attention_9_query_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOpUassignvariableop_72_adam_transformer_block_9_multi_head_self_attention_9_query_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOpUassignvariableop_73_adam_transformer_block_9_multi_head_self_attention_9_key_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOpSassignvariableop_74_adam_transformer_block_9_multi_head_self_attention_9_key_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOpWassignvariableop_75_adam_transformer_block_9_multi_head_self_attention_9_value_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOpUassignvariableop_76_adam_transformer_block_9_multi_head_self_attention_9_value_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOpUassignvariableop_77_adam_transformer_block_9_multi_head_self_attention_9_out_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOpSassignvariableop_78_adam_transformer_block_9_multi_head_self_attention_9_out_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_dense_36_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_dense_36_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_dense_37_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_dense_37_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOpKassignvariableop_83_adam_transformer_block_9_layer_normalization_32_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOpJassignvariableop_84_adam_transformer_block_9_layer_normalization_32_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOpKassignvariableop_85_adam_transformer_block_9_layer_normalization_33_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOpJassignvariableop_86_adam_transformer_block_9_layer_normalization_33_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_87Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_88IdentityIdentity_87:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_88Identity_88:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_259210

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????p:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
??
?
C__inference_model_4_layer_call_and_return_conditional_losses_258016

inputs<
*dense_35_tensordot_readvariableop_resource:d@6
(dense_35_biasadd_readvariableop_resource:@B
4layer_normalization_31_mul_3_readvariableop_resource:@@
2layer_normalization_31_add_readvariableop_resource:@i
Wtransformer_block_9_multi_head_self_attention_9_query_tensordot_readvariableop_resource:@c
Utransformer_block_9_multi_head_self_attention_9_query_biasadd_readvariableop_resource:g
Utransformer_block_9_multi_head_self_attention_9_key_tensordot_readvariableop_resource:@a
Stransformer_block_9_multi_head_self_attention_9_key_biasadd_readvariableop_resource:i
Wtransformer_block_9_multi_head_self_attention_9_value_tensordot_readvariableop_resource:@c
Utransformer_block_9_multi_head_self_attention_9_value_biasadd_readvariableop_resource:g
Utransformer_block_9_multi_head_self_attention_9_out_tensordot_readvariableop_resource:a
Stransformer_block_9_multi_head_self_attention_9_out_biasadd_readvariableop_resource:^
Ptransformer_block_9_layer_normalization_32_batchnorm_mul_readvariableop_resource:Z
Ltransformer_block_9_layer_normalization_32_batchnorm_readvariableop_resource:]
Ktransformer_block_9_sequential_9_dense_36_tensordot_readvariableop_resource:W
Itransformer_block_9_sequential_9_dense_36_biasadd_readvariableop_resource:]
Ktransformer_block_9_sequential_9_dense_37_tensordot_readvariableop_resource:W
Itransformer_block_9_sequential_9_dense_37_biasadd_readvariableop_resource:^
Ptransformer_block_9_layer_normalization_33_batchnorm_mul_readvariableop_resource:Z
Ltransformer_block_9_layer_normalization_33_batchnorm_readvariableop_resource::
'dense_38_matmul_readvariableop_resource:	?6
(dense_38_biasadd_readvariableop_resource:B
4layer_normalization_34_mul_2_readvariableop_resource:@
2layer_normalization_34_add_readvariableop_resource:9
'dense_39_matmul_readvariableop_resource:6
(dense_39_biasadd_readvariableop_resource:
identity??dense_35/BiasAdd/ReadVariableOp?!dense_35/Tensordot/ReadVariableOp?1dense_35/kernel/Regularizer/Square/ReadVariableOp?dense_38/BiasAdd/ReadVariableOp?dense_38/MatMul/ReadVariableOp?1dense_38/kernel/Regularizer/Square/ReadVariableOp?dense_39/BiasAdd/ReadVariableOp?dense_39/MatMul/ReadVariableOp?1dense_39/kernel/Regularizer/Square/ReadVariableOp?)layer_normalization_31/add/ReadVariableOp?+layer_normalization_31/mul_3/ReadVariableOp?)layer_normalization_34/add/ReadVariableOp?+layer_normalization_34/mul_2/ReadVariableOp?Ctransformer_block_9/layer_normalization_32/batchnorm/ReadVariableOp?Gtransformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOp?Ctransformer_block_9/layer_normalization_33/batchnorm/ReadVariableOp?Gtransformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOp?Jtransformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOp?Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOp?Jtransformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOp?Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOp?Ltransformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOp?Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOp?Ltransformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOp?Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOp?@transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOp?Btransformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOp?@transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOp?Btransformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp?
!dense_35/Tensordot/ReadVariableOpReadVariableOp*dense_35_tensordot_readvariableop_resource*
_output_shapes

:d@*
dtype0a
dense_35/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_35/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_35/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_35/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_35/Tensordot/GatherV2GatherV2!dense_35/Tensordot/Shape:output:0 dense_35/Tensordot/free:output:0)dense_35/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_35/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_35/Tensordot/GatherV2_1GatherV2!dense_35/Tensordot/Shape:output:0 dense_35/Tensordot/axes:output:0+dense_35/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_35/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_35/Tensordot/ProdProd$dense_35/Tensordot/GatherV2:output:0!dense_35/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_35/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_35/Tensordot/Prod_1Prod&dense_35/Tensordot/GatherV2_1:output:0#dense_35/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_35/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_35/Tensordot/concatConcatV2 dense_35/Tensordot/free:output:0 dense_35/Tensordot/axes:output:0'dense_35/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_35/Tensordot/stackPack dense_35/Tensordot/Prod:output:0"dense_35/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_35/Tensordot/transpose	Transposeinputs"dense_35/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????pd?
dense_35/Tensordot/ReshapeReshape dense_35/Tensordot/transpose:y:0!dense_35/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_35/Tensordot/MatMulMatMul#dense_35/Tensordot/Reshape:output:0)dense_35/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
dense_35/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@b
 dense_35/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_35/Tensordot/concat_1ConcatV2$dense_35/Tensordot/GatherV2:output:0#dense_35/Tensordot/Const_2:output:0)dense_35/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_35/TensordotReshape#dense_35/Tensordot/MatMul:product:0$dense_35/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p@?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_35/BiasAddBiasAdddense_35/Tensordot:output:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@f
dense_35/ReluReludense_35/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p@r
dropout_27/IdentityIdentitydense_35/Relu:activations:0*
T0*+
_output_shapes
:?????????p@h
layer_normalization_31/ShapeShapedropout_27/Identity:output:0*
T0*
_output_shapes
:t
*layer_normalization_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_31/strided_sliceStridedSlice%layer_normalization_31/Shape:output:03layer_normalization_31/strided_slice/stack:output:05layer_normalization_31/strided_slice/stack_1:output:05layer_normalization_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_31/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_31/mulMul%layer_normalization_31/mul/x:output:0-layer_normalization_31/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_31/strided_slice_1StridedSlice%layer_normalization_31/Shape:output:05layer_normalization_31/strided_slice_1/stack:output:07layer_normalization_31/strided_slice_1/stack_1:output:07layer_normalization_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_31/mul_1Mullayer_normalization_31/mul:z:0/layer_normalization_31/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_31/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_31/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_31/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_31/strided_slice_2StridedSlice%layer_normalization_31/Shape:output:05layer_normalization_31/strided_slice_2/stack:output:07layer_normalization_31/strided_slice_2/stack_1:output:07layer_normalization_31/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_31/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_31/mul_2Mul'layer_normalization_31/mul_2/x:output:0/layer_normalization_31/strided_slice_2:output:0*
T0*
_output_shapes
: h
&layer_normalization_31/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_31/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_31/Reshape/shapePack/layer_normalization_31/Reshape/shape/0:output:0 layer_normalization_31/mul_1:z:0 layer_normalization_31/mul_2:z:0/layer_normalization_31/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_31/ReshapeReshapedropout_27/Identity:output:0-layer_normalization_31/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@z
"layer_normalization_31/ones/packedPack layer_normalization_31/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_31/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_31/onesFill+layer_normalization_31/ones/packed:output:0*layer_normalization_31/ones/Const:output:0*
T0*#
_output_shapes
:?????????{
#layer_normalization_31/zeros/packedPack layer_normalization_31/mul_1:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_31/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_31/zerosFill,layer_normalization_31/zeros/packed:output:0+layer_normalization_31/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_31/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_31/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_31/FusedBatchNormV3FusedBatchNormV3'layer_normalization_31/Reshape:output:0$layer_normalization_31/ones:output:0%layer_normalization_31/zeros:output:0%layer_normalization_31/Const:output:0'layer_normalization_31/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
 layer_normalization_31/Reshape_1Reshape+layer_normalization_31/FusedBatchNormV3:y:0%layer_normalization_31/Shape:output:0*
T0*+
_output_shapes
:?????????p@?
+layer_normalization_31/mul_3/ReadVariableOpReadVariableOp4layer_normalization_31_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization_31/mul_3Mul)layer_normalization_31/Reshape_1:output:03layer_normalization_31/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@?
)layer_normalization_31/add/ReadVariableOpReadVariableOp2layer_normalization_31_add_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization_31/addAddV2 layer_normalization_31/mul_3:z:01layer_normalization_31/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@?
5transformer_block_9/multi_head_self_attention_9/ShapeShapelayer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Ctransformer_block_9/multi_head_self_attention_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Etransformer_block_9/multi_head_self_attention_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Etransformer_block_9/multi_head_self_attention_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=transformer_block_9/multi_head_self_attention_9/strided_sliceStridedSlice>transformer_block_9/multi_head_self_attention_9/Shape:output:0Ltransformer_block_9/multi_head_self_attention_9/strided_slice/stack:output:0Ntransformer_block_9/multi_head_self_attention_9/strided_slice/stack_1:output:0Ntransformer_block_9/multi_head_self_attention_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOpReadVariableOpWtransformer_block_9_multi_head_self_attention_9_query_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0?
Dtransformer_block_9/multi_head_self_attention_9/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Dtransformer_block_9/multi_head_self_attention_9/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Etransformer_block_9/multi_head_self_attention_9/query/Tensordot/ShapeShapelayer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2GatherV2Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/Shape:output:0Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/free:output:0Vtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Otransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Jtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1GatherV2Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/Shape:output:0Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/axes:output:0Xtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Etransformer_block_9/multi_head_self_attention_9/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_9/multi_head_self_attention_9/query/Tensordot/ProdProdQtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2:output:0Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Gtransformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Ftransformer_block_9/multi_head_self_attention_9/query/Tensordot/Prod_1ProdStransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2_1:output:0Ptransformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Ktransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/query/Tensordot/concatConcatV2Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/free:output:0Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/axes:output:0Ttransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Etransformer_block_9/multi_head_self_attention_9/query/Tensordot/stackPackMtransformer_block_9/multi_head_self_attention_9/query/Tensordot/Prod:output:0Otransformer_block_9/multi_head_self_attention_9/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Itransformer_block_9/multi_head_self_attention_9/query/Tensordot/transpose	Transposelayer_normalization_31/add:z:0Otransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
Gtransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReshapeReshapeMtransformer_block_9/multi_head_self_attention_9/query/Tensordot/transpose:y:0Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Ftransformer_block_9/multi_head_self_attention_9/query/Tensordot/MatMulMatMulPtransformer_block_9/multi_head_self_attention_9/query/Tensordot/Reshape:output:0Vtransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Gtransformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Mtransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1ConcatV2Qtransformer_block_9/multi_head_self_attention_9/query/Tensordot/GatherV2:output:0Ptransformer_block_9/multi_head_self_attention_9/query/Tensordot/Const_2:output:0Vtransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
?transformer_block_9/multi_head_self_attention_9/query/TensordotReshapePtransformer_block_9/multi_head_self_attention_9/query/Tensordot/MatMul:product:0Qtransformer_block_9/multi_head_self_attention_9/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Ltransformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOpReadVariableOpUtransformer_block_9_multi_head_self_attention_9_query_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
=transformer_block_9/multi_head_self_attention_9/query/BiasAddBiasAddHtransformer_block_9/multi_head_self_attention_9/query/Tensordot:output:0Ttransformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_9_multi_head_self_attention_9_key_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0?
Btransformer_block_9/multi_head_self_attention_9/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_9/multi_head_self_attention_9/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_9/multi_head_self_attention_9/key/Tensordot/ShapeShapelayer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2GatherV2Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/Shape:output:0Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/free:output:0Ttransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1GatherV2Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/Shape:output:0Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/axes:output:0Vtransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_9/multi_head_self_attention_9/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_9/multi_head_self_attention_9/key/Tensordot/ProdProdOtransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2:output:0Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_9/multi_head_self_attention_9/key/Tensordot/Prod_1ProdQtransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2_1:output:0Ntransformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_9/multi_head_self_attention_9/key/Tensordot/concatConcatV2Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/free:output:0Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/axes:output:0Rtransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_9/multi_head_self_attention_9/key/Tensordot/stackPackKtransformer_block_9/multi_head_self_attention_9/key/Tensordot/Prod:output:0Mtransformer_block_9/multi_head_self_attention_9/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_9/multi_head_self_attention_9/key/Tensordot/transpose	Transposelayer_normalization_31/add:z:0Mtransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
Etransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReshapeReshapeKtransformer_block_9/multi_head_self_attention_9/key/Tensordot/transpose:y:0Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_9/multi_head_self_attention_9/key/Tensordot/MatMulMatMulNtransformer_block_9/multi_head_self_attention_9/key/Tensordot/Reshape:output:0Ttransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1ConcatV2Otransformer_block_9/multi_head_self_attention_9/key/Tensordot/GatherV2:output:0Ntransformer_block_9/multi_head_self_attention_9/key/Tensordot/Const_2:output:0Ttransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_9/multi_head_self_attention_9/key/TensordotReshapeNtransformer_block_9/multi_head_self_attention_9/key/Tensordot/MatMul:product:0Otransformer_block_9/multi_head_self_attention_9/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Jtransformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_9_multi_head_self_attention_9_key_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_9/multi_head_self_attention_9/key/BiasAddBiasAddFtransformer_block_9/multi_head_self_attention_9/key/Tensordot:output:0Rtransformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOpReadVariableOpWtransformer_block_9_multi_head_self_attention_9_value_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0?
Dtransformer_block_9/multi_head_self_attention_9/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Dtransformer_block_9/multi_head_self_attention_9/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Etransformer_block_9/multi_head_self_attention_9/value/Tensordot/ShapeShapelayer_normalization_31/add:z:0*
T0*
_output_shapes
:?
Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2GatherV2Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/Shape:output:0Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/free:output:0Vtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Otransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Jtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1GatherV2Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/Shape:output:0Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/axes:output:0Xtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Etransformer_block_9/multi_head_self_attention_9/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_9/multi_head_self_attention_9/value/Tensordot/ProdProdQtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2:output:0Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Gtransformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Ftransformer_block_9/multi_head_self_attention_9/value/Tensordot/Prod_1ProdStransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2_1:output:0Ptransformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Ktransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/value/Tensordot/concatConcatV2Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/free:output:0Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/axes:output:0Ttransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Etransformer_block_9/multi_head_self_attention_9/value/Tensordot/stackPackMtransformer_block_9/multi_head_self_attention_9/value/Tensordot/Prod:output:0Otransformer_block_9/multi_head_self_attention_9/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Itransformer_block_9/multi_head_self_attention_9/value/Tensordot/transpose	Transposelayer_normalization_31/add:z:0Otransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
Gtransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReshapeReshapeMtransformer_block_9/multi_head_self_attention_9/value/Tensordot/transpose:y:0Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Ftransformer_block_9/multi_head_self_attention_9/value/Tensordot/MatMulMatMulPtransformer_block_9/multi_head_self_attention_9/value/Tensordot/Reshape:output:0Vtransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Gtransformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Mtransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1ConcatV2Qtransformer_block_9/multi_head_self_attention_9/value/Tensordot/GatherV2:output:0Ptransformer_block_9/multi_head_self_attention_9/value/Tensordot/Const_2:output:0Vtransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
?transformer_block_9/multi_head_self_attention_9/value/TensordotReshapePtransformer_block_9/multi_head_self_attention_9/value/Tensordot/MatMul:product:0Qtransformer_block_9/multi_head_self_attention_9/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Ltransformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOpReadVariableOpUtransformer_block_9_multi_head_self_attention_9_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
=transformer_block_9/multi_head_self_attention_9/value/BiasAddBiasAddHtransformer_block_9/multi_head_self_attention_9/value/Tensordot:output:0Ttransformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
?transformer_block_9/multi_head_self_attention_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
?transformer_block_9/multi_head_self_attention_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
?transformer_block_9/multi_head_self_attention_9/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
=transformer_block_9/multi_head_self_attention_9/Reshape/shapePackFtransformer_block_9/multi_head_self_attention_9/strided_slice:output:0Htransformer_block_9/multi_head_self_attention_9/Reshape/shape/1:output:0Htransformer_block_9/multi_head_self_attention_9/Reshape/shape/2:output:0Htransformer_block_9/multi_head_self_attention_9/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
7transformer_block_9/multi_head_self_attention_9/ReshapeReshapeFtransformer_block_9/multi_head_self_attention_9/query/BiasAdd:output:0Ftransformer_block_9/multi_head_self_attention_9/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????p?
>transformer_block_9/multi_head_self_attention_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
9transformer_block_9/multi_head_self_attention_9/transpose	Transpose@transformer_block_9/multi_head_self_attention_9/Reshape:output:0Gtransformer_block_9/multi_head_self_attention_9/transpose/perm:output:0*
T0*/
_output_shapes
:?????????p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Atransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
?transformer_block_9/multi_head_self_attention_9/Reshape_1/shapePackFtransformer_block_9/multi_head_self_attention_9/strided_slice:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/1:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/2:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_9/multi_head_self_attention_9/Reshape_1ReshapeDtransformer_block_9/multi_head_self_attention_9/key/BiasAdd:output:0Htransformer_block_9/multi_head_self_attention_9/Reshape_1/shape:output:0*
T0*/
_output_shapes
:?????????p?
@transformer_block_9/multi_head_self_attention_9/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
;transformer_block_9/multi_head_self_attention_9/transpose_1	TransposeBtransformer_block_9/multi_head_self_attention_9/Reshape_1:output:0Itransformer_block_9/multi_head_self_attention_9/transpose_1/perm:output:0*
T0*/
_output_shapes
:?????????p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Atransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
?transformer_block_9/multi_head_self_attention_9/Reshape_2/shapePackFtransformer_block_9/multi_head_self_attention_9/strided_slice:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/1:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/2:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_9/multi_head_self_attention_9/Reshape_2ReshapeFtransformer_block_9/multi_head_self_attention_9/value/BiasAdd:output:0Htransformer_block_9/multi_head_self_attention_9/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????p?
@transformer_block_9/multi_head_self_attention_9/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
;transformer_block_9/multi_head_self_attention_9/transpose_2	TransposeBtransformer_block_9/multi_head_self_attention_9/Reshape_2:output:0Itransformer_block_9/multi_head_self_attention_9/transpose_2/perm:output:0*
T0*/
_output_shapes
:?????????p?
6transformer_block_9/multi_head_self_attention_9/MatMulBatchMatMulV2=transformer_block_9/multi_head_self_attention_9/transpose:y:0?transformer_block_9/multi_head_self_attention_9/transpose_1:y:0*
T0*/
_output_shapes
:?????????pp*
adj_y(?
7transformer_block_9/multi_head_self_attention_9/Shape_1Shape?transformer_block_9/multi_head_self_attention_9/transpose_1:y:0*
T0*
_output_shapes
:?
Etransformer_block_9/multi_head_self_attention_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Gtransformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Gtransformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?transformer_block_9/multi_head_self_attention_9/strided_slice_1StridedSlice@transformer_block_9/multi_head_self_attention_9/Shape_1:output:0Ntransformer_block_9/multi_head_self_attention_9/strided_slice_1/stack:output:0Ptransformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_1:output:0Ptransformer_block_9/multi_head_self_attention_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
4transformer_block_9/multi_head_self_attention_9/CastCastHtransformer_block_9/multi_head_self_attention_9/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
4transformer_block_9/multi_head_self_attention_9/SqrtSqrt8transformer_block_9/multi_head_self_attention_9/Cast:y:0*
T0*
_output_shapes
: ?
7transformer_block_9/multi_head_self_attention_9/truedivRealDiv?transformer_block_9/multi_head_self_attention_9/MatMul:output:08transformer_block_9/multi_head_self_attention_9/Sqrt:y:0*
T0*/
_output_shapes
:?????????pp?
7transformer_block_9/multi_head_self_attention_9/SoftmaxSoftmax;transformer_block_9/multi_head_self_attention_9/truediv:z:0*
T0*/
_output_shapes
:?????????pp?
8transformer_block_9/multi_head_self_attention_9/MatMul_1BatchMatMulV2Atransformer_block_9/multi_head_self_attention_9/Softmax:softmax:0?transformer_block_9/multi_head_self_attention_9/transpose_2:y:0*
T0*/
_output_shapes
:?????????p?
@transformer_block_9/multi_head_self_attention_9/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
;transformer_block_9/multi_head_self_attention_9/transpose_3	TransposeAtransformer_block_9/multi_head_self_attention_9/MatMul_1:output:0Itransformer_block_9/multi_head_self_attention_9/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p?
Atransformer_block_9/multi_head_self_attention_9/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
?transformer_block_9/multi_head_self_attention_9/Reshape_3/shapePackFtransformer_block_9/multi_head_self_attention_9/strided_slice:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_3/shape/1:output:0Jtransformer_block_9/multi_head_self_attention_9/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_9/multi_head_self_attention_9/Reshape_3Reshape?transformer_block_9/multi_head_self_attention_9/transpose_3:y:0Htransformer_block_9/multi_head_self_attention_9/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????p?
Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_9_multi_head_self_attention_9_out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Btransformer_block_9/multi_head_self_attention_9/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_9/multi_head_self_attention_9/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_9/multi_head_self_attention_9/out/Tensordot/ShapeShapeBtransformer_block_9/multi_head_self_attention_9/Reshape_3:output:0*
T0*
_output_shapes
:?
Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2GatherV2Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/Shape:output:0Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/free:output:0Ttransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1GatherV2Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/Shape:output:0Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/axes:output:0Vtransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_9/multi_head_self_attention_9/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_9/multi_head_self_attention_9/out/Tensordot/ProdProdOtransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2:output:0Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_9/multi_head_self_attention_9/out/Tensordot/Prod_1ProdQtransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2_1:output:0Ntransformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_9/multi_head_self_attention_9/out/Tensordot/concatConcatV2Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/free:output:0Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/axes:output:0Rtransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_9/multi_head_self_attention_9/out/Tensordot/stackPackKtransformer_block_9/multi_head_self_attention_9/out/Tensordot/Prod:output:0Mtransformer_block_9/multi_head_self_attention_9/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_9/multi_head_self_attention_9/out/Tensordot/transpose	TransposeBtransformer_block_9/multi_head_self_attention_9/Reshape_3:output:0Mtransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
Etransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReshapeReshapeKtransformer_block_9/multi_head_self_attention_9/out/Tensordot/transpose:y:0Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_9/multi_head_self_attention_9/out/Tensordot/MatMulMatMulNtransformer_block_9/multi_head_self_attention_9/out/Tensordot/Reshape:output:0Ttransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1ConcatV2Otransformer_block_9/multi_head_self_attention_9/out/Tensordot/GatherV2:output:0Ntransformer_block_9/multi_head_self_attention_9/out/Tensordot/Const_2:output:0Ttransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_9/multi_head_self_attention_9/out/TensordotReshapeNtransformer_block_9/multi_head_self_attention_9/out/Tensordot/MatMul:product:0Otransformer_block_9/multi_head_self_attention_9/out/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
Jtransformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_9_multi_head_self_attention_9_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_9/multi_head_self_attention_9/out/BiasAddBiasAddFtransformer_block_9/multi_head_self_attention_9/out/Tensordot:output:0Rtransformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
'transformer_block_9/dropout_28/IdentityIdentityDtransformer_block_9/multi_head_self_attention_9/out/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
Itransformer_block_9/layer_normalization_32/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_block_9/layer_normalization_32/moments/meanMean0transformer_block_9/dropout_28/Identity:output:0Rtransformer_block_9/layer_normalization_32/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
?transformer_block_9/layer_normalization_32/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_32/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
Dtransformer_block_9/layer_normalization_32/moments/SquaredDifferenceSquaredDifference0transformer_block_9/dropout_28/Identity:output:0Htransformer_block_9/layer_normalization_32/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
Mtransformer_block_9/layer_normalization_32/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_9/layer_normalization_32/moments/varianceMeanHtransformer_block_9/layer_normalization_32/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_32/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(
:transformer_block_9/layer_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
8transformer_block_9/layer_normalization_32/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_32/moments/variance:output:0Ctransformer_block_9/layer_normalization_32/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_32/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_32/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
Gtransformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_block_9/layer_normalization_32/batchnorm/mulMul>transformer_block_9/layer_normalization_32/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_32/batchnorm/mul_1Mul0transformer_block_9/dropout_28/Identity:output:0<transformer_block_9/layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_32/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_32/moments/mean:output:0<transformer_block_9/layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
Ctransformer_block_9/layer_normalization_32/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_block_9/layer_normalization_32/batchnorm/subSubKtransformer_block_9/layer_normalization_32/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_32/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_32/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_32/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_32/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p?
Btransformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_36_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
8transformer_block_9/sequential_9/dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_block_9/sequential_9/dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
9transformer_block_9/sequential_9/dense_36/Tensordot/ShapeShape>transformer_block_9/layer_normalization_32/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
Atransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_36/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_36/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_36/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_36/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
9transformer_block_9/sequential_9/dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
8transformer_block_9/sequential_9/dense_36/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
;transformer_block_9/sequential_9/dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_9/sequential_9/dense_36/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
?transformer_block_9/sequential_9/dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:transformer_block_9/sequential_9/dense_36/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_36/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_36/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_9/sequential_9/dense_36/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_36/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_9/sequential_9/dense_36/Tensordot/transpose	Transpose>transformer_block_9/layer_normalization_32/batchnorm/add_1:z:0Ctransformer_block_9/sequential_9/dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
;transformer_block_9/sequential_9/dense_36/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_36/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
:transformer_block_9/sequential_9/dense_36/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_36/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;transformer_block_9/sequential_9/dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_9/sequential_9/dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_9/sequential_9/dense_36/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_36/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_36/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_9/sequential_9/dense_36/TensordotReshapeDtransformer_block_9/sequential_9/dense_36/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
@transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1transformer_block_9/sequential_9/dense_36/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_36/Tensordot:output:0Htransformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
.transformer_block_9/sequential_9/dense_36/ReluRelu:transformer_block_9/sequential_9/dense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
Btransformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_37_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
8transformer_block_9/sequential_9/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_block_9/sequential_9/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
9transformer_block_9/sequential_9/dense_37/Tensordot/ShapeShape<transformer_block_9/sequential_9/dense_36/Relu:activations:0*
T0*
_output_shapes
:?
Atransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_37/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_37/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_37/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_37/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
9transformer_block_9/sequential_9/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
8transformer_block_9/sequential_9/dense_37/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
;transformer_block_9/sequential_9/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_9/sequential_9/dense_37/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
?transformer_block_9/sequential_9/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:transformer_block_9/sequential_9/dense_37/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_37/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_37/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_9/sequential_9/dense_37/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_37/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_9/sequential_9/dense_37/Tensordot/transpose	Transpose<transformer_block_9/sequential_9/dense_36/Relu:activations:0Ctransformer_block_9/sequential_9/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
;transformer_block_9/sequential_9/dense_37/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_37/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
:transformer_block_9/sequential_9/dense_37/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_37/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;transformer_block_9/sequential_9/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_9/sequential_9/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_9/sequential_9/dense_37/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_37/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_37/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_9/sequential_9/dense_37/TensordotReshapeDtransformer_block_9/sequential_9/dense_37/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
@transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1transformer_block_9/sequential_9/dense_37/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_37/Tensordot:output:0Htransformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
'transformer_block_9/dropout_29/IdentityIdentity:transformer_block_9/sequential_9/dense_37/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
transformer_block_9/addAddV2>transformer_block_9/layer_normalization_32/batchnorm/add_1:z:00transformer_block_9/dropout_29/Identity:output:0*
T0*+
_output_shapes
:?????????p?
Itransformer_block_9/layer_normalization_33/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_block_9/layer_normalization_33/moments/meanMeantransformer_block_9/add:z:0Rtransformer_block_9/layer_normalization_33/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
?transformer_block_9/layer_normalization_33/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_33/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
Dtransformer_block_9/layer_normalization_33/moments/SquaredDifferenceSquaredDifferencetransformer_block_9/add:z:0Htransformer_block_9/layer_normalization_33/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
Mtransformer_block_9/layer_normalization_33/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_9/layer_normalization_33/moments/varianceMeanHtransformer_block_9/layer_normalization_33/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_33/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(
:transformer_block_9/layer_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
8transformer_block_9/layer_normalization_33/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_33/moments/variance:output:0Ctransformer_block_9/layer_normalization_33/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_33/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_33/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
Gtransformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_block_9/layer_normalization_33/batchnorm/mulMul>transformer_block_9/layer_normalization_33/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_33/batchnorm/mul_1Multransformer_block_9/add:z:0<transformer_block_9/layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_33/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_33/moments/mean:output:0<transformer_block_9/layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
Ctransformer_block_9/layer_normalization_33/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_block_9/layer_normalization_33/batchnorm/subSubKtransformer_block_9/layer_normalization_33/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_33/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
:transformer_block_9/layer_normalization_33/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_33/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_33/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_4/ReshapeReshape>transformer_block_9/layer_normalization_33/batchnorm/add_1:z:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????n
dropout_30/IdentityIdentityflatten_4/Reshape:output:0*
T0*(
_output_shapes
:???????????
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_38/MatMulMatMuldropout_30/Identity:output:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
layer_normalization_34/ShapeShapedense_38/Relu:activations:0*
T0*
_output_shapes
:t
*layer_normalization_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_34/strided_sliceStridedSlice%layer_normalization_34/Shape:output:03layer_normalization_34/strided_slice/stack:output:05layer_normalization_34/strided_slice/stack_1:output:05layer_normalization_34/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_34/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_34/mulMul%layer_normalization_34/mul/x:output:0-layer_normalization_34/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_34/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_34/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_34/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_34/strided_slice_1StridedSlice%layer_normalization_34/Shape:output:05layer_normalization_34/strided_slice_1/stack:output:07layer_normalization_34/strided_slice_1/stack_1:output:07layer_normalization_34/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_34/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_34/mul_1Mul'layer_normalization_34/mul_1/x:output:0/layer_normalization_34/strided_slice_1:output:0*
T0*
_output_shapes
: h
&layer_normalization_34/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_34/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_34/Reshape/shapePack/layer_normalization_34/Reshape/shape/0:output:0layer_normalization_34/mul:z:0 layer_normalization_34/mul_1:z:0/layer_normalization_34/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_34/ReshapeReshapedense_38/Relu:activations:0-layer_normalization_34/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????x
"layer_normalization_34/ones/packedPacklayer_normalization_34/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_34/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_34/onesFill+layer_normalization_34/ones/packed:output:0*layer_normalization_34/ones/Const:output:0*
T0*#
_output_shapes
:?????????y
#layer_normalization_34/zeros/packedPacklayer_normalization_34/mul:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_34/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_34/zerosFill,layer_normalization_34/zeros/packed:output:0+layer_normalization_34/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_34/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_34/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_34/FusedBatchNormV3FusedBatchNormV3'layer_normalization_34/Reshape:output:0$layer_normalization_34/ones:output:0%layer_normalization_34/zeros:output:0%layer_normalization_34/Const:output:0'layer_normalization_34/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
 layer_normalization_34/Reshape_1Reshape+layer_normalization_34/FusedBatchNormV3:y:0%layer_normalization_34/Shape:output:0*
T0*'
_output_shapes
:??????????
+layer_normalization_34/mul_2/ReadVariableOpReadVariableOp4layer_normalization_34_mul_2_readvariableop_resource*
_output_shapes
:*
dtype0?
layer_normalization_34/mul_2Mul)layer_normalization_34/Reshape_1:output:03layer_normalization_34/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)layer_normalization_34/add/ReadVariableOpReadVariableOp2layer_normalization_34_add_readvariableop_resource*
_output_shapes
:*
dtype0?
layer_normalization_34/addAddV2 layer_normalization_34/mul_2:z:01layer_normalization_34/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_39/MatMulMatMullayer_normalization_34/add:z:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_39/ReluReludense_39/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_35_tensordot_readvariableop_resource*
_output_shapes

:d@*
dtype0?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d@r
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_35/kernel/Regularizer/SumSum&dense_35/kernel/Regularizer/Square:y:0*dense_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?r
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentitydense_39/Relu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_35/BiasAdd/ReadVariableOp"^dense_35/Tensordot/ReadVariableOp2^dense_35/kernel/Regularizer/Square/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp2^dense_39/kernel/Regularizer/Square/ReadVariableOp*^layer_normalization_31/add/ReadVariableOp,^layer_normalization_31/mul_3/ReadVariableOp*^layer_normalization_34/add/ReadVariableOp,^layer_normalization_34/mul_2/ReadVariableOpD^transformer_block_9/layer_normalization_32/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOpD^transformer_block_9/layer_normalization_33/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOpK^transformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOpM^transformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOpK^transformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOpM^transformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOpM^transformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOpO^transformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOpM^transformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOpO^transformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOpA^transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOpA^transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2F
!dense_35/Tensordot/ReadVariableOp!dense_35/Tensordot/ReadVariableOp2f
1dense_35/kernel/Regularizer/Square/ReadVariableOp1dense_35/kernel/Regularizer/Square/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2f
1dense_39/kernel/Regularizer/Square/ReadVariableOp1dense_39/kernel/Regularizer/Square/ReadVariableOp2V
)layer_normalization_31/add/ReadVariableOp)layer_normalization_31/add/ReadVariableOp2Z
+layer_normalization_31/mul_3/ReadVariableOp+layer_normalization_31/mul_3/ReadVariableOp2V
)layer_normalization_34/add/ReadVariableOp)layer_normalization_34/add/ReadVariableOp2Z
+layer_normalization_34/mul_2/ReadVariableOp+layer_normalization_34/mul_2/ReadVariableOp2?
Ctransformer_block_9/layer_normalization_32/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_32/batchnorm/ReadVariableOp2?
Gtransformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_32/batchnorm/mul/ReadVariableOp2?
Ctransformer_block_9/layer_normalization_33/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_33/batchnorm/ReadVariableOp2?
Gtransformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_33/batchnorm/mul/ReadVariableOp2?
Jtransformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOpJtransformer_block_9/multi_head_self_attention_9/key/BiasAdd/ReadVariableOp2?
Ltransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOpLtransformer_block_9/multi_head_self_attention_9/key/Tensordot/ReadVariableOp2?
Jtransformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOpJtransformer_block_9/multi_head_self_attention_9/out/BiasAdd/ReadVariableOp2?
Ltransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOpLtransformer_block_9/multi_head_self_attention_9/out/Tensordot/ReadVariableOp2?
Ltransformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOpLtransformer_block_9/multi_head_self_attention_9/query/BiasAdd/ReadVariableOp2?
Ntransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOpNtransformer_block_9/multi_head_self_attention_9/query/Tensordot/ReadVariableOp2?
Ltransformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOpLtransformer_block_9/multi_head_self_attention_9/value/BiasAdd/ReadVariableOp2?
Ntransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOpNtransformer_block_9/multi_head_self_attention_9/value/Tensordot/ReadVariableOp2?
@transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_36/BiasAdd/ReadVariableOp2?
Btransformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_36/Tensordot/ReadVariableOp2?
@transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_37/BiasAdd/ReadVariableOp2?
Btransformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_37/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????pd
 
_user_specified_nameinputs
?
d
F__inference_dropout_27_layer_call_and_return_conditional_losses_258557

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????p@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????p@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????p@:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
?
)__inference_dense_37_layer_call_fn_259574

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_255948s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????p: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
??
?
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_256993

inputsU
Cmulti_head_self_attention_9_query_tensordot_readvariableop_resource:@O
Amulti_head_self_attention_9_query_biasadd_readvariableop_resource:S
Amulti_head_self_attention_9_key_tensordot_readvariableop_resource:@M
?multi_head_self_attention_9_key_biasadd_readvariableop_resource:U
Cmulti_head_self_attention_9_value_tensordot_readvariableop_resource:@O
Amulti_head_self_attention_9_value_biasadd_readvariableop_resource:S
Amulti_head_self_attention_9_out_tensordot_readvariableop_resource:M
?multi_head_self_attention_9_out_biasadd_readvariableop_resource:J
<layer_normalization_32_batchnorm_mul_readvariableop_resource:F
8layer_normalization_32_batchnorm_readvariableop_resource:I
7sequential_9_dense_36_tensordot_readvariableop_resource:C
5sequential_9_dense_36_biasadd_readvariableop_resource:I
7sequential_9_dense_37_tensordot_readvariableop_resource:C
5sequential_9_dense_37_biasadd_readvariableop_resource:J
<layer_normalization_33_batchnorm_mul_readvariableop_resource:F
8layer_normalization_33_batchnorm_readvariableop_resource:
identity??/layer_normalization_32/batchnorm/ReadVariableOp?3layer_normalization_32/batchnorm/mul/ReadVariableOp?/layer_normalization_33/batchnorm/ReadVariableOp?3layer_normalization_33/batchnorm/mul/ReadVariableOp?6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp?8multi_head_self_attention_9/key/Tensordot/ReadVariableOp?6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp?8multi_head_self_attention_9/out/Tensordot/ReadVariableOp?8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp?:multi_head_self_attention_9/query/Tensordot/ReadVariableOp?8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp?:multi_head_self_attention_9/value/Tensordot/ReadVariableOp?,sequential_9/dense_36/BiasAdd/ReadVariableOp?.sequential_9/dense_36/Tensordot/ReadVariableOp?,sequential_9/dense_37/BiasAdd/ReadVariableOp?.sequential_9/dense_37/Tensordot/ReadVariableOpW
!multi_head_self_attention_9/ShapeShapeinputs*
T0*
_output_shapes
:y
/multi_head_self_attention_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1multi_head_self_attention_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1multi_head_self_attention_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)multi_head_self_attention_9/strided_sliceStridedSlice*multi_head_self_attention_9/Shape:output:08multi_head_self_attention_9/strided_slice/stack:output:0:multi_head_self_attention_9/strided_slice/stack_1:output:0:multi_head_self_attention_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:multi_head_self_attention_9/query/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_9_query_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0z
0multi_head_self_attention_9/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0multi_head_self_attention_9/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
1multi_head_self_attention_9/query/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention_9/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/query/Tensordot/GatherV2GatherV2:multi_head_self_attention_9/query/Tensordot/Shape:output:09multi_head_self_attention_9/query/Tensordot/free:output:0Bmulti_head_self_attention_9/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;multi_head_self_attention_9/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6multi_head_self_attention_9/query/Tensordot/GatherV2_1GatherV2:multi_head_self_attention_9/query/Tensordot/Shape:output:09multi_head_self_attention_9/query/Tensordot/axes:output:0Dmulti_head_self_attention_9/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1multi_head_self_attention_9/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/query/Tensordot/ProdProd=multi_head_self_attention_9/query/Tensordot/GatherV2:output:0:multi_head_self_attention_9/query/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention_9/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2multi_head_self_attention_9/query/Tensordot/Prod_1Prod?multi_head_self_attention_9/query/Tensordot/GatherV2_1:output:0<multi_head_self_attention_9/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention_9/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/query/Tensordot/concatConcatV29multi_head_self_attention_9/query/Tensordot/free:output:09multi_head_self_attention_9/query/Tensordot/axes:output:0@multi_head_self_attention_9/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1multi_head_self_attention_9/query/Tensordot/stackPack9multi_head_self_attention_9/query/Tensordot/Prod:output:0;multi_head_self_attention_9/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5multi_head_self_attention_9/query/Tensordot/transpose	Transposeinputs;multi_head_self_attention_9/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
3multi_head_self_attention_9/query/Tensordot/ReshapeReshape9multi_head_self_attention_9/query/Tensordot/transpose:y:0:multi_head_self_attention_9/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2multi_head_self_attention_9/query/Tensordot/MatMulMatMul<multi_head_self_attention_9/query/Tensordot/Reshape:output:0Bmulti_head_self_attention_9/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
3multi_head_self_attention_9/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:{
9multi_head_self_attention_9/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/query/Tensordot/concat_1ConcatV2=multi_head_self_attention_9/query/Tensordot/GatherV2:output:0<multi_head_self_attention_9/query/Tensordot/Const_2:output:0Bmulti_head_self_attention_9/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+multi_head_self_attention_9/query/TensordotReshape<multi_head_self_attention_9/query/Tensordot/MatMul:product:0=multi_head_self_attention_9/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/query/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_query_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_self_attention_9/query/BiasAddBiasAdd4multi_head_self_attention_9/query/Tensordot:output:0@multi_head_self_attention_9/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/key/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_key_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0x
.multi_head_self_attention_9/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention_9/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_self_attention_9/key/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_self_attention_9/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/key/Tensordot/GatherV2GatherV28multi_head_self_attention_9/key/Tensordot/Shape:output:07multi_head_self_attention_9/key/Tensordot/free:output:0@multi_head_self_attention_9/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention_9/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/key/Tensordot/GatherV2_1GatherV28multi_head_self_attention_9/key/Tensordot/Shape:output:07multi_head_self_attention_9/key/Tensordot/axes:output:0Bmulti_head_self_attention_9/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention_9/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_self_attention_9/key/Tensordot/ProdProd;multi_head_self_attention_9/key/Tensordot/GatherV2:output:08multi_head_self_attention_9/key/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention_9/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/key/Tensordot/Prod_1Prod=multi_head_self_attention_9/key/Tensordot/GatherV2_1:output:0:multi_head_self_attention_9/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention_9/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_self_attention_9/key/Tensordot/concatConcatV27multi_head_self_attention_9/key/Tensordot/free:output:07multi_head_self_attention_9/key/Tensordot/axes:output:0>multi_head_self_attention_9/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_self_attention_9/key/Tensordot/stackPack7multi_head_self_attention_9/key/Tensordot/Prod:output:09multi_head_self_attention_9/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_self_attention_9/key/Tensordot/transpose	Transposeinputs9multi_head_self_attention_9/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
1multi_head_self_attention_9/key/Tensordot/ReshapeReshape7multi_head_self_attention_9/key/Tensordot/transpose:y:08multi_head_self_attention_9/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_self_attention_9/key/Tensordot/MatMulMatMul:multi_head_self_attention_9/key/Tensordot/Reshape:output:0@multi_head_self_attention_9/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_self_attention_9/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_self_attention_9/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/key/Tensordot/concat_1ConcatV2;multi_head_self_attention_9/key/Tensordot/GatherV2:output:0:multi_head_self_attention_9/key/Tensordot/Const_2:output:0@multi_head_self_attention_9/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_self_attention_9/key/TensordotReshape:multi_head_self_attention_9/key/Tensordot/MatMul:product:0;multi_head_self_attention_9/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
6multi_head_self_attention_9/key/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_9_key_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_self_attention_9/key/BiasAddBiasAdd2multi_head_self_attention_9/key/Tensordot:output:0>multi_head_self_attention_9/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
:multi_head_self_attention_9/value/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_9_value_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0z
0multi_head_self_attention_9/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0multi_head_self_attention_9/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
1multi_head_self_attention_9/value/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention_9/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/value/Tensordot/GatherV2GatherV2:multi_head_self_attention_9/value/Tensordot/Shape:output:09multi_head_self_attention_9/value/Tensordot/free:output:0Bmulti_head_self_attention_9/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;multi_head_self_attention_9/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6multi_head_self_attention_9/value/Tensordot/GatherV2_1GatherV2:multi_head_self_attention_9/value/Tensordot/Shape:output:09multi_head_self_attention_9/value/Tensordot/axes:output:0Dmulti_head_self_attention_9/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1multi_head_self_attention_9/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/value/Tensordot/ProdProd=multi_head_self_attention_9/value/Tensordot/GatherV2:output:0:multi_head_self_attention_9/value/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention_9/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2multi_head_self_attention_9/value/Tensordot/Prod_1Prod?multi_head_self_attention_9/value/Tensordot/GatherV2_1:output:0<multi_head_self_attention_9/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention_9/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/value/Tensordot/concatConcatV29multi_head_self_attention_9/value/Tensordot/free:output:09multi_head_self_attention_9/value/Tensordot/axes:output:0@multi_head_self_attention_9/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1multi_head_self_attention_9/value/Tensordot/stackPack9multi_head_self_attention_9/value/Tensordot/Prod:output:0;multi_head_self_attention_9/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5multi_head_self_attention_9/value/Tensordot/transpose	Transposeinputs;multi_head_self_attention_9/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p@?
3multi_head_self_attention_9/value/Tensordot/ReshapeReshape9multi_head_self_attention_9/value/Tensordot/transpose:y:0:multi_head_self_attention_9/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2multi_head_self_attention_9/value/Tensordot/MatMulMatMul<multi_head_self_attention_9/value/Tensordot/Reshape:output:0Bmulti_head_self_attention_9/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
3multi_head_self_attention_9/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:{
9multi_head_self_attention_9/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/value/Tensordot/concat_1ConcatV2=multi_head_self_attention_9/value/Tensordot/GatherV2:output:0<multi_head_self_attention_9/value/Tensordot/Const_2:output:0Bmulti_head_self_attention_9/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+multi_head_self_attention_9/value/TensordotReshape<multi_head_self_attention_9/value/Tensordot/MatMul:product:0=multi_head_self_attention_9/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/value/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_self_attention_9/value/BiasAddBiasAdd4multi_head_self_attention_9/value/Tensordot:output:0@multi_head_self_attention_9/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pm
+multi_head_self_attention_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :pm
+multi_head_self_attention_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention_9/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
)multi_head_self_attention_9/Reshape/shapePack2multi_head_self_attention_9/strided_slice:output:04multi_head_self_attention_9/Reshape/shape/1:output:04multi_head_self_attention_9/Reshape/shape/2:output:04multi_head_self_attention_9/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
#multi_head_self_attention_9/ReshapeReshape2multi_head_self_attention_9/query/BiasAdd:output:02multi_head_self_attention_9/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????p?
*multi_head_self_attention_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
%multi_head_self_attention_9/transpose	Transpose,multi_head_self_attention_9/Reshape:output:03multi_head_self_attention_9/transpose/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-multi_head_self_attention_9/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_1/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_1/shape/1:output:06multi_head_self_attention_9/Reshape_1/shape/2:output:06multi_head_self_attention_9/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_1Reshape0multi_head_self_attention_9/key/BiasAdd:output:04multi_head_self_attention_9/Reshape_1/shape:output:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_1	Transpose.multi_head_self_attention_9/Reshape_1:output:05multi_head_self_attention_9/transpose_1/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-multi_head_self_attention_9/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_2/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_2/shape/1:output:06multi_head_self_attention_9/Reshape_2/shape/2:output:06multi_head_self_attention_9/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_2Reshape2multi_head_self_attention_9/value/BiasAdd:output:04multi_head_self_attention_9/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_2	Transpose.multi_head_self_attention_9/Reshape_2:output:05multi_head_self_attention_9/transpose_2/perm:output:0*
T0*/
_output_shapes
:?????????p?
"multi_head_self_attention_9/MatMulBatchMatMulV2)multi_head_self_attention_9/transpose:y:0+multi_head_self_attention_9/transpose_1:y:0*
T0*/
_output_shapes
:?????????pp*
adj_y(~
#multi_head_self_attention_9/Shape_1Shape+multi_head_self_attention_9/transpose_1:y:0*
T0*
_output_shapes
:?
1multi_head_self_attention_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
3multi_head_self_attention_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3multi_head_self_attention_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+multi_head_self_attention_9/strided_slice_1StridedSlice,multi_head_self_attention_9/Shape_1:output:0:multi_head_self_attention_9/strided_slice_1/stack:output:0<multi_head_self_attention_9/strided_slice_1/stack_1:output:0<multi_head_self_attention_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
 multi_head_self_attention_9/CastCast4multi_head_self_attention_9/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: o
 multi_head_self_attention_9/SqrtSqrt$multi_head_self_attention_9/Cast:y:0*
T0*
_output_shapes
: ?
#multi_head_self_attention_9/truedivRealDiv+multi_head_self_attention_9/MatMul:output:0$multi_head_self_attention_9/Sqrt:y:0*
T0*/
_output_shapes
:?????????pp?
#multi_head_self_attention_9/SoftmaxSoftmax'multi_head_self_attention_9/truediv:z:0*
T0*/
_output_shapes
:?????????pp?
$multi_head_self_attention_9/MatMul_1BatchMatMulV2-multi_head_self_attention_9/Softmax:softmax:0+multi_head_self_attention_9/transpose_2:y:0*
T0*/
_output_shapes
:?????????p?
,multi_head_self_attention_9/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
'multi_head_self_attention_9/transpose_3	Transpose-multi_head_self_attention_9/MatMul_1:output:05multi_head_self_attention_9/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????po
-multi_head_self_attention_9/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :po
-multi_head_self_attention_9/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
+multi_head_self_attention_9/Reshape_3/shapePack2multi_head_self_attention_9/strided_slice:output:06multi_head_self_attention_9/Reshape_3/shape/1:output:06multi_head_self_attention_9/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
%multi_head_self_attention_9/Reshape_3Reshape+multi_head_self_attention_9/transpose_3:y:04multi_head_self_attention_9/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????p?
8multi_head_self_attention_9/out/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_9_out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_self_attention_9/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention_9/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
/multi_head_self_attention_9/out/Tensordot/ShapeShape.multi_head_self_attention_9/Reshape_3:output:0*
T0*
_output_shapes
:y
7multi_head_self_attention_9/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/out/Tensordot/GatherV2GatherV28multi_head_self_attention_9/out/Tensordot/Shape:output:07multi_head_self_attention_9/out/Tensordot/free:output:0@multi_head_self_attention_9/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention_9/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_self_attention_9/out/Tensordot/GatherV2_1GatherV28multi_head_self_attention_9/out/Tensordot/Shape:output:07multi_head_self_attention_9/out/Tensordot/axes:output:0Bmulti_head_self_attention_9/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention_9/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_self_attention_9/out/Tensordot/ProdProd;multi_head_self_attention_9/out/Tensordot/GatherV2:output:08multi_head_self_attention_9/out/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention_9/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_self_attention_9/out/Tensordot/Prod_1Prod=multi_head_self_attention_9/out/Tensordot/GatherV2_1:output:0:multi_head_self_attention_9/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention_9/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_self_attention_9/out/Tensordot/concatConcatV27multi_head_self_attention_9/out/Tensordot/free:output:07multi_head_self_attention_9/out/Tensordot/axes:output:0>multi_head_self_attention_9/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_self_attention_9/out/Tensordot/stackPack7multi_head_self_attention_9/out/Tensordot/Prod:output:09multi_head_self_attention_9/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_self_attention_9/out/Tensordot/transpose	Transpose.multi_head_self_attention_9/Reshape_3:output:09multi_head_self_attention_9/out/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
1multi_head_self_attention_9/out/Tensordot/ReshapeReshape7multi_head_self_attention_9/out/Tensordot/transpose:y:08multi_head_self_attention_9/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_self_attention_9/out/Tensordot/MatMulMatMul:multi_head_self_attention_9/out/Tensordot/Reshape:output:0@multi_head_self_attention_9/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_self_attention_9/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_self_attention_9/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_self_attention_9/out/Tensordot/concat_1ConcatV2;multi_head_self_attention_9/out/Tensordot/GatherV2:output:0:multi_head_self_attention_9/out/Tensordot/Const_2:output:0@multi_head_self_attention_9/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_self_attention_9/out/TensordotReshape:multi_head_self_attention_9/out/Tensordot/MatMul:product:0;multi_head_self_attention_9/out/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
6multi_head_self_attention_9/out/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_9_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_self_attention_9/out/BiasAddBiasAdd2multi_head_self_attention_9/out/Tensordot:output:0>multi_head_self_attention_9/out/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p]
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_28/dropout/MulMul0multi_head_self_attention_9/out/BiasAdd:output:0!dropout_28/dropout/Const:output:0*
T0*+
_output_shapes
:?????????px
dropout_28/dropout/ShapeShape0multi_head_self_attention_9/out/BiasAdd:output:0*
T0*
_output_shapes
:?
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????p*
dtype0f
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????p?
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????p?
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????p
5layer_normalization_32/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_32/moments/meanMeandropout_28/dropout/Mul_1:z:0>layer_normalization_32/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
+layer_normalization_32/moments/StopGradientStopGradient,layer_normalization_32/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
0layer_normalization_32/moments/SquaredDifferenceSquaredDifferencedropout_28/dropout/Mul_1:z:04layer_normalization_32/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
9layer_normalization_32/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
'layer_normalization_32/moments/varianceMean4layer_normalization_32/moments/SquaredDifference:z:0Blayer_normalization_32/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(k
&layer_normalization_32/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
$layer_normalization_32/batchnorm/addAddV20layer_normalization_32/moments/variance:output:0/layer_normalization_32/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/RsqrtRsqrt(layer_normalization_32/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
3layer_normalization_32/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_32_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_32/batchnorm/mulMul*layer_normalization_32/batchnorm/Rsqrt:y:0;layer_normalization_32/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/mul_1Muldropout_28/dropout/Mul_1:z:0(layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/mul_2Mul,layer_normalization_32/moments/mean:output:0(layer_normalization_32/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
/layer_normalization_32/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_32_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_32/batchnorm/subSub7layer_normalization_32/batchnorm/ReadVariableOp:value:0*layer_normalization_32/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_32/batchnorm/add_1AddV2*layer_normalization_32/batchnorm/mul_1:z:0(layer_normalization_32/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p?
.sequential_9/dense_36/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_36_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_9/dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_9/dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_9/dense_36/Tensordot/ShapeShape*layer_normalization_32/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_9/dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_36/Tensordot/GatherV2GatherV2.sequential_9/dense_36/Tensordot/Shape:output:0-sequential_9/dense_36/Tensordot/free:output:06sequential_9/dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_9/dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_9/dense_36/Tensordot/GatherV2_1GatherV2.sequential_9/dense_36/Tensordot/Shape:output:0-sequential_9/dense_36/Tensordot/axes:output:08sequential_9/dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_9/dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_9/dense_36/Tensordot/ProdProd1sequential_9/dense_36/Tensordot/GatherV2:output:0.sequential_9/dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_9/dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_9/dense_36/Tensordot/Prod_1Prod3sequential_9/dense_36/Tensordot/GatherV2_1:output:00sequential_9/dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_9/dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_9/dense_36/Tensordot/concatConcatV2-sequential_9/dense_36/Tensordot/free:output:0-sequential_9/dense_36/Tensordot/axes:output:04sequential_9/dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_9/dense_36/Tensordot/stackPack-sequential_9/dense_36/Tensordot/Prod:output:0/sequential_9/dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_9/dense_36/Tensordot/transpose	Transpose*layer_normalization_32/batchnorm/add_1:z:0/sequential_9/dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
'sequential_9/dense_36/Tensordot/ReshapeReshape-sequential_9/dense_36/Tensordot/transpose:y:0.sequential_9/dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_9/dense_36/Tensordot/MatMulMatMul0sequential_9/dense_36/Tensordot/Reshape:output:06sequential_9/dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_9/dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_9/dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_36/Tensordot/concat_1ConcatV21sequential_9/dense_36/Tensordot/GatherV2:output:00sequential_9/dense_36/Tensordot/Const_2:output:06sequential_9/dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_9/dense_36/TensordotReshape0sequential_9/dense_36/Tensordot/MatMul:product:01sequential_9/dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/dense_36/BiasAddBiasAdd(sequential_9/dense_36/Tensordot:output:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
.sequential_9/dense_37/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_37_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_9/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_9/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_9/dense_37/Tensordot/ShapeShape(sequential_9/dense_36/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_9/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_37/Tensordot/GatherV2GatherV2.sequential_9/dense_37/Tensordot/Shape:output:0-sequential_9/dense_37/Tensordot/free:output:06sequential_9/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_9/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_9/dense_37/Tensordot/GatherV2_1GatherV2.sequential_9/dense_37/Tensordot/Shape:output:0-sequential_9/dense_37/Tensordot/axes:output:08sequential_9/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_9/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_9/dense_37/Tensordot/ProdProd1sequential_9/dense_37/Tensordot/GatherV2:output:0.sequential_9/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_9/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_9/dense_37/Tensordot/Prod_1Prod3sequential_9/dense_37/Tensordot/GatherV2_1:output:00sequential_9/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_9/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_9/dense_37/Tensordot/concatConcatV2-sequential_9/dense_37/Tensordot/free:output:0-sequential_9/dense_37/Tensordot/axes:output:04sequential_9/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_9/dense_37/Tensordot/stackPack-sequential_9/dense_37/Tensordot/Prod:output:0/sequential_9/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_9/dense_37/Tensordot/transpose	Transpose(sequential_9/dense_36/Relu:activations:0/sequential_9/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
'sequential_9/dense_37/Tensordot/ReshapeReshape-sequential_9/dense_37/Tensordot/transpose:y:0.sequential_9/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_9/dense_37/Tensordot/MatMulMatMul0sequential_9/dense_37/Tensordot/Reshape:output:06sequential_9/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_9/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_9/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_9/dense_37/Tensordot/concat_1ConcatV21sequential_9/dense_37/Tensordot/GatherV2:output:00sequential_9/dense_37/Tensordot/Const_2:output:06sequential_9/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_9/dense_37/TensordotReshape0sequential_9/dense_37/Tensordot/MatMul:product:01sequential_9/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/dense_37/BiasAddBiasAdd(sequential_9/dense_37/Tensordot:output:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p]
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_29/dropout/MulMul&sequential_9/dense_37/BiasAdd:output:0!dropout_29/dropout/Const:output:0*
T0*+
_output_shapes
:?????????pn
dropout_29/dropout/ShapeShape&sequential_9/dense_37/BiasAdd:output:0*
T0*
_output_shapes
:?
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????p*
dtype0f
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????p?
dropout_29/dropout/CastCast#dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????p?
dropout_29/dropout/Mul_1Muldropout_29/dropout/Mul:z:0dropout_29/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????p?
addAddV2*layer_normalization_32/batchnorm/add_1:z:0dropout_29/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????p
5layer_normalization_33/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_33/moments/meanMeanadd:z:0>layer_normalization_33/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(?
+layer_normalization_33/moments/StopGradientStopGradient,layer_normalization_33/moments/mean:output:0*
T0*+
_output_shapes
:?????????p?
0layer_normalization_33/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_33/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????p?
9layer_normalization_33/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
'layer_normalization_33/moments/varianceMean4layer_normalization_33/moments/SquaredDifference:z:0Blayer_normalization_33/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????p*
	keep_dims(k
&layer_normalization_33/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
$layer_normalization_33/batchnorm/addAddV20layer_normalization_33/moments/variance:output:0/layer_normalization_33/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/RsqrtRsqrt(layer_normalization_33/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????p?
3layer_normalization_33/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_33_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_33/batchnorm/mulMul*layer_normalization_33/batchnorm/Rsqrt:y:0;layer_normalization_33/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/mul_1Muladd:z:0(layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/mul_2Mul,layer_normalization_33/moments/mean:output:0(layer_normalization_33/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????p?
/layer_normalization_33/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_33_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
$layer_normalization_33/batchnorm/subSub7layer_normalization_33/batchnorm/ReadVariableOp:value:0*layer_normalization_33/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????p?
&layer_normalization_33/batchnorm/add_1AddV2*layer_normalization_33/batchnorm/mul_1:z:0(layer_normalization_33/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????p}
IdentityIdentity*layer_normalization_33/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp0^layer_normalization_32/batchnorm/ReadVariableOp4^layer_normalization_32/batchnorm/mul/ReadVariableOp0^layer_normalization_33/batchnorm/ReadVariableOp4^layer_normalization_33/batchnorm/mul/ReadVariableOp7^multi_head_self_attention_9/key/BiasAdd/ReadVariableOp9^multi_head_self_attention_9/key/Tensordot/ReadVariableOp7^multi_head_self_attention_9/out/BiasAdd/ReadVariableOp9^multi_head_self_attention_9/out/Tensordot/ReadVariableOp9^multi_head_self_attention_9/query/BiasAdd/ReadVariableOp;^multi_head_self_attention_9/query/Tensordot/ReadVariableOp9^multi_head_self_attention_9/value/BiasAdd/ReadVariableOp;^multi_head_self_attention_9/value/Tensordot/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp/^sequential_9/dense_36/Tensordot/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp/^sequential_9/dense_37/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????p@: : : : : : : : : : : : : : : : 2b
/layer_normalization_32/batchnorm/ReadVariableOp/layer_normalization_32/batchnorm/ReadVariableOp2j
3layer_normalization_32/batchnorm/mul/ReadVariableOp3layer_normalization_32/batchnorm/mul/ReadVariableOp2b
/layer_normalization_33/batchnorm/ReadVariableOp/layer_normalization_33/batchnorm/ReadVariableOp2j
3layer_normalization_33/batchnorm/mul/ReadVariableOp3layer_normalization_33/batchnorm/mul/ReadVariableOp2p
6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp6multi_head_self_attention_9/key/BiasAdd/ReadVariableOp2t
8multi_head_self_attention_9/key/Tensordot/ReadVariableOp8multi_head_self_attention_9/key/Tensordot/ReadVariableOp2p
6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp6multi_head_self_attention_9/out/BiasAdd/ReadVariableOp2t
8multi_head_self_attention_9/out/Tensordot/ReadVariableOp8multi_head_self_attention_9/out/Tensordot/ReadVariableOp2t
8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp8multi_head_self_attention_9/query/BiasAdd/ReadVariableOp2x
:multi_head_self_attention_9/query/Tensordot/ReadVariableOp:multi_head_self_attention_9/query/Tensordot/ReadVariableOp2t
8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp8multi_head_self_attention_9/value/BiasAdd/ReadVariableOp2x
:multi_head_self_attention_9/value/Tensordot/ReadVariableOp:multi_head_self_attention_9/value/Tensordot/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2`
.sequential_9/dense_36/Tensordot/ReadVariableOp.sequential_9/dense_36/Tensordot/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2`
.sequential_9/dense_37/Tensordot/ReadVariableOp.sequential_9/dense_37/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
?
D__inference_dense_36_layer_call_and_return_conditional_losses_259565

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pT
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????pe
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????pz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????p: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_256039
dense_36_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_256015s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????p: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:?????????p
(
_user_specified_namedense_36_input
?$
?
R__inference_layer_normalization_31_layer_call_and_return_conditional_losses_256171

inputs+
mul_3_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:?????????p@n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:@*
dtype0t
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0i
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????p@r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????p@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_256460

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????p:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
d
+__inference_dropout_27_layer_call_fn_258552

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_257058s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????p@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_259363L
:dense_35_kernel_regularizer_square_readvariableop_resource:d@
identity??1dense_35/kernel/Regularizer/Square/ReadVariableOp?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_35_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:d@*
dtype0?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d@r
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_35/kernel/Regularizer/SumSum&dense_35/kernel/Regularizer/Square:y:0*dense_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_35/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_35/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_35/kernel/Regularizer/Square/ReadVariableOp1dense_35/kernel/Regularizer/Square/ReadVariableOp
?

e
F__inference_dropout_27_layer_call_and_return_conditional_losses_257058

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????p@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????p@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????p@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????p@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????p@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????p@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????p@:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_259468

inputs<
*dense_36_tensordot_readvariableop_resource:6
(dense_36_biasadd_readvariableop_resource:<
*dense_37_tensordot_readvariableop_resource:6
(dense_37_biasadd_readvariableop_resource:
identity??dense_36/BiasAdd/ReadVariableOp?!dense_36/Tensordot/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?!dense_37/Tensordot/ReadVariableOp?
!dense_36/Tensordot/ReadVariableOpReadVariableOp*dense_36_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_36/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_36/Tensordot/GatherV2GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/free:output:0)dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_36/Tensordot/GatherV2_1GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/axes:output:0+dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_36/Tensordot/ProdProd$dense_36/Tensordot/GatherV2:output:0!dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_36/Tensordot/Prod_1Prod&dense_36/Tensordot/GatherV2_1:output:0#dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_36/Tensordot/concatConcatV2 dense_36/Tensordot/free:output:0 dense_36/Tensordot/axes:output:0'dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_36/Tensordot/stackPack dense_36/Tensordot/Prod:output:0"dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_36/Tensordot/transpose	Transposeinputs"dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
dense_36/Tensordot/ReshapeReshape dense_36/Tensordot/transpose:y:0!dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_36/Tensordot/MatMulMatMul#dense_36/Tensordot/Reshape:output:0)dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_36/Tensordot/concat_1ConcatV2$dense_36/Tensordot/GatherV2:output:0#dense_36/Tensordot/Const_2:output:0)dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_36/TensordotReshape#dense_36/Tensordot/MatMul:product:0$dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_36/BiasAddBiasAdddense_36/Tensordot:output:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pf
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????p?
!dense_37/Tensordot/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_37/Tensordot/ShapeShapedense_36/Relu:activations:0*
T0*
_output_shapes
:b
 dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_37/Tensordot/GatherV2GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/free:output:0)dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_37/Tensordot/GatherV2_1GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/axes:output:0+dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_37/Tensordot/ProdProd$dense_37/Tensordot/GatherV2:output:0!dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_37/Tensordot/Prod_1Prod&dense_37/Tensordot/GatherV2_1:output:0#dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_37/Tensordot/concatConcatV2 dense_37/Tensordot/free:output:0 dense_37/Tensordot/axes:output:0'dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_37/Tensordot/stackPack dense_37/Tensordot/Prod:output:0"dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_37/Tensordot/transpose	Transposedense_36/Relu:activations:0"dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????p?
dense_37/Tensordot/ReshapeReshape dense_37/Tensordot/transpose:y:0!dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_37/Tensordot/MatMulMatMul#dense_37/Tensordot/Reshape:output:0)dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_37/Tensordot/concat_1ConcatV2$dense_37/Tensordot/GatherV2:output:0#dense_37/Tensordot/Const_2:output:0)dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_37/TensordotReshape#dense_37/Tensordot/MatMul:product:0$dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_37/BiasAddBiasAdddense_37/Tensordot:output:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????pl
IdentityIdentitydense_37/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????p?
NoOpNoOp ^dense_36/BiasAdd/ReadVariableOp"^dense_36/Tensordot/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp"^dense_37/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????p: : : : 2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2F
!dense_36/Tensordot/ReadVariableOp!dense_36/Tensordot/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2F
!dense_37/Tensordot/ReadVariableOp!dense_37/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????p
 
_user_specified_nameinputs
?
?
4__inference_transformer_block_9_layer_call_fn_258662

inputs
unknown:@
	unknown_0:
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????p*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_256420s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????p@: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
?
(__inference_model_4_layer_call_fn_256637
input_10
unknown:d@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:	?

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_256582o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????pd
"
_user_specified_name
input_10
?%
?
D__inference_dense_35_layer_call_and_return_conditional_losses_256111

inputs3
!tensordot_readvariableop_resource:d@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?1dense_35/kernel/Regularizer/Square/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:d@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????pd?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????p@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????p@T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????p@?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:d@*
dtype0?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d@r
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_35/kernel/Regularizer/SumSum&dense_35/kernel/Regularizer/Square:y:0*dense_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????p@?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp2^dense_35/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????pd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2f
1dense_35/kernel/Regularizer/Square/ReadVariableOp1dense_35/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????pd
 
_user_specified_nameinputs
?
?
D__inference_dense_38_layer_call_and_return_conditional_losses_256486

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_38/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?r
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_model_4_layer_call_fn_257572

inputs
unknown:d@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:	?

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_256582o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????pd: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????pd
 
_user_specified_nameinputs
?
d
F__inference_dropout_27_layer_call_and_return_conditional_losses_256122

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????p@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????p@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????p@:S O
+
_output_shapes
:?????????p@
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_259385L
:dense_39_kernel_regularizer_square_readvariableop_resource:
identity??1dense_39/kernel/Regularizer/Square/ReadVariableOp?
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_39_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0?
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_39/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_39/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_39/kernel/Regularizer/Square/ReadVariableOp1dense_39/kernel/Regularizer/Square/ReadVariableOp"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
input_105
serving_default_input_10:0?????????pd<
dense_390
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
 _random_generator
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#axis
	$gamma
%beta
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
?
,att
-ffn
.
layernorm1
/
layernorm2
0dropout1
1dropout2
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
?
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B_random_generator
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Maxis
	Ngamma
Obeta
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
?
^iter

_beta_1

`beta_2
	adecay
blearning_ratem?m?$m?%m?Em?Fm?Nm?Om?Vm?Wm?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?om?pm?qm?rm?v?v?$v?%v?Ev?Fv?Nv?Ov?Vv?Wv?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?"
	optimizer
?
0
1
$2
%3
c4
d5
e6
f7
g8
h9
i10
j11
k12
l13
m14
n15
o16
p17
q18
r19
E20
F21
N22
O23
V24
W25"
trackable_list_wrapper
?
0
1
$2
%3
c4
d5
e6
f7
g8
h9
i10
j11
k12
l13
m14
n15
o16
p17
q18
r19
E20
F21
N22
O23
V24
W25"
trackable_list_wrapper
5
s0
t1
u2"
trackable_list_wrapper
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_model_4_layer_call_fn_256637
(__inference_model_4_layer_call_fn_257572
(__inference_model_4_layer_call_fn_257629
(__inference_model_4_layer_call_fn_257325?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_4_layer_call_and_return_conditional_losses_258016
C__inference_model_4_layer_call_and_return_conditional_losses_258431
C__inference_model_4_layer_call_and_return_conditional_losses_257408
C__inference_model_4_layer_call_and_return_conditional_losses_257491?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_255874input_10"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
{serving_default"
signature_map
!:d@2dense_35/kernel
:@2dense_35/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
s0"
trackable_list_wrapper
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_35_layer_call_fn_258505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_35_layer_call_and_return_conditional_losses_258542?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
+__inference_dropout_27_layer_call_fn_258547
+__inference_dropout_27_layer_call_fn_258552?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_27_layer_call_and_return_conditional_losses_258557
F__inference_dropout_27_layer_call_and_return_conditional_losses_258569?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
*:(@2layer_normalization_31/gamma
):'@2layer_normalization_31/beta
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?2?
7__inference_layer_normalization_31_layer_call_fn_258578?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_layer_normalization_31_layer_call_and_return_conditional_losses_258625?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?query_dense
?	key_dense
?value_dense
?combine_heads
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
	?axis
	ogamma
pbeta
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	qgamma
rbeta
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15"
trackable_list_wrapper
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
?2?
4__inference_transformer_block_9_layer_call_fn_258662
4__inference_transformer_block_9_layer_call_fn_258699?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_258942
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_259199?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_flatten_4_layer_call_fn_259204?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_4_layer_call_and_return_conditional_losses_259210?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
>	variables
?trainable_variables
@regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
+__inference_dropout_30_layer_call_fn_259215
+__inference_dropout_30_layer_call_fn_259220?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_30_layer_call_and_return_conditional_losses_259225
F__inference_dropout_30_layer_call_and_return_conditional_losses_259237?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
": 	?2dense_38/kernel
:2dense_38/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
'
t0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_38_layer_call_fn_259252?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_38_layer_call_and_return_conditional_losses_259269?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
*:(2layer_normalization_34/gamma
):'2layer_normalization_34/beta
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
?2?
7__inference_layer_normalization_34_layer_call_fn_259278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_layer_normalization_34_layer_call_and_return_conditional_losses_259320?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
!:2dense_39/kernel
:2dense_39/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
'
u0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_39_layer_call_fn_259335?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_39_layer_call_and_return_conditional_losses_259352?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
N:L@2<transformer_block_9/multi_head_self_attention_9/query/kernel
H:F2:transformer_block_9/multi_head_self_attention_9/query/bias
L:J@2:transformer_block_9/multi_head_self_attention_9/key/kernel
F:D28transformer_block_9/multi_head_self_attention_9/key/bias
N:L@2<transformer_block_9/multi_head_self_attention_9/value/kernel
H:F2:transformer_block_9/multi_head_self_attention_9/value/bias
L:J2:transformer_block_9/multi_head_self_attention_9/out/kernel
F:D28transformer_block_9/multi_head_self_attention_9/out/bias
!:2dense_36/kernel
:2dense_36/bias
!:2dense_37/kernel
:2dense_37/bias
>:<20transformer_block_9/layer_normalization_32/gamma
=:;2/transformer_block_9/layer_normalization_32/beta
>:<20transformer_block_9/layer_normalization_33/gamma
=:;2/transformer_block_9/layer_normalization_33/beta
?2?
__inference_loss_fn_0_259363?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_259374?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_259385?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_258490input_10"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
s0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

ckernel
dbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ekernel
fbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

gkernel
hbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ikernel
jbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
X
c0
d1
e2
f3
g4
h5
i6
j7"
trackable_list_wrapper
X
c0
d1
e2
f3
g4
h5
i6
j7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?

kkernel
lbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

mkernel
nbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
<
k0
l1
m2
n3"
trackable_list_wrapper
<
k0
l1
m2
n3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_sequential_9_layer_call_fn_255966
-__inference_sequential_9_layer_call_fn_259398
-__inference_sequential_9_layer_call_fn_259411
-__inference_sequential_9_layer_call_fn_256039?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_9_layer_call_and_return_conditional_losses_259468
H__inference_sequential_9_layer_call_and_return_conditional_losses_259525
H__inference_sequential_9_layer_call_and_return_conditional_losses_256053
H__inference_sequential_9_layer_call_and_return_conditional_losses_256067?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
t0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
u0"
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_36_layer_call_fn_259534?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_36_layer_call_and_return_conditional_losses_259565?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_37_layer_call_fn_259574?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_37_layer_call_and_return_conditional_losses_259604?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
&:$d@2Adam/dense_35/kernel/m
 :@2Adam/dense_35/bias/m
/:-@2#Adam/layer_normalization_31/gamma/m
.:,@2"Adam/layer_normalization_31/beta/m
':%	?2Adam/dense_38/kernel/m
 :2Adam/dense_38/bias/m
/:-2#Adam/layer_normalization_34/gamma/m
.:,2"Adam/layer_normalization_34/beta/m
&:$2Adam/dense_39/kernel/m
 :2Adam/dense_39/bias/m
S:Q@2CAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/m
M:K2AAdam/transformer_block_9/multi_head_self_attention_9/query/bias/m
Q:O@2AAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/m
K:I2?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/m
S:Q@2CAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/m
M:K2AAdam/transformer_block_9/multi_head_self_attention_9/value/bias/m
Q:O2AAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/m
K:I2?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/m
&:$2Adam/dense_36/kernel/m
 :2Adam/dense_36/bias/m
&:$2Adam/dense_37/kernel/m
 :2Adam/dense_37/bias/m
C:A27Adam/transformer_block_9/layer_normalization_32/gamma/m
B:@26Adam/transformer_block_9/layer_normalization_32/beta/m
C:A27Adam/transformer_block_9/layer_normalization_33/gamma/m
B:@26Adam/transformer_block_9/layer_normalization_33/beta/m
&:$d@2Adam/dense_35/kernel/v
 :@2Adam/dense_35/bias/v
/:-@2#Adam/layer_normalization_31/gamma/v
.:,@2"Adam/layer_normalization_31/beta/v
':%	?2Adam/dense_38/kernel/v
 :2Adam/dense_38/bias/v
/:-2#Adam/layer_normalization_34/gamma/v
.:,2"Adam/layer_normalization_34/beta/v
&:$2Adam/dense_39/kernel/v
 :2Adam/dense_39/bias/v
S:Q@2CAdam/transformer_block_9/multi_head_self_attention_9/query/kernel/v
M:K2AAdam/transformer_block_9/multi_head_self_attention_9/query/bias/v
Q:O@2AAdam/transformer_block_9/multi_head_self_attention_9/key/kernel/v
K:I2?Adam/transformer_block_9/multi_head_self_attention_9/key/bias/v
S:Q@2CAdam/transformer_block_9/multi_head_self_attention_9/value/kernel/v
M:K2AAdam/transformer_block_9/multi_head_self_attention_9/value/bias/v
Q:O2AAdam/transformer_block_9/multi_head_self_attention_9/out/kernel/v
K:I2?Adam/transformer_block_9/multi_head_self_attention_9/out/bias/v
&:$2Adam/dense_36/kernel/v
 :2Adam/dense_36/bias/v
&:$2Adam/dense_37/kernel/v
 :2Adam/dense_37/bias/v
C:A27Adam/transformer_block_9/layer_normalization_32/gamma/v
B:@26Adam/transformer_block_9/layer_normalization_32/beta/v
C:A27Adam/transformer_block_9/layer_normalization_33/gamma/v
B:@26Adam/transformer_block_9/layer_normalization_33/beta/v?
!__inference__wrapped_model_255874?$%cdefghijopklmnqrEFNOVW5?2
+?(
&?#
input_10?????????pd
? "3?0
.
dense_39"?
dense_39??????????
D__inference_dense_35_layer_call_and_return_conditional_losses_258542d3?0
)?&
$?!
inputs?????????pd
? ")?&
?
0?????????p@
? ?
)__inference_dense_35_layer_call_fn_258505W3?0
)?&
$?!
inputs?????????pd
? "??????????p@?
D__inference_dense_36_layer_call_and_return_conditional_losses_259565dkl3?0
)?&
$?!
inputs?????????p
? ")?&
?
0?????????p
? ?
)__inference_dense_36_layer_call_fn_259534Wkl3?0
)?&
$?!
inputs?????????p
? "??????????p?
D__inference_dense_37_layer_call_and_return_conditional_losses_259604dmn3?0
)?&
$?!
inputs?????????p
? ")?&
?
0?????????p
? ?
)__inference_dense_37_layer_call_fn_259574Wmn3?0
)?&
$?!
inputs?????????p
? "??????????p?
D__inference_dense_38_layer_call_and_return_conditional_losses_259269]EF0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_38_layer_call_fn_259252PEF0?-
&?#
!?
inputs??????????
? "???????????
D__inference_dense_39_layer_call_and_return_conditional_losses_259352\VW/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_39_layer_call_fn_259335OVW/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_dropout_27_layer_call_and_return_conditional_losses_258557d7?4
-?*
$?!
inputs?????????p@
p 
? ")?&
?
0?????????p@
? ?
F__inference_dropout_27_layer_call_and_return_conditional_losses_258569d7?4
-?*
$?!
inputs?????????p@
p
? ")?&
?
0?????????p@
? ?
+__inference_dropout_27_layer_call_fn_258547W7?4
-?*
$?!
inputs?????????p@
p 
? "??????????p@?
+__inference_dropout_27_layer_call_fn_258552W7?4
-?*
$?!
inputs?????????p@
p
? "??????????p@?
F__inference_dropout_30_layer_call_and_return_conditional_losses_259225^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
F__inference_dropout_30_layer_call_and_return_conditional_losses_259237^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
+__inference_dropout_30_layer_call_fn_259215Q4?1
*?'
!?
inputs??????????
p 
? "????????????
+__inference_dropout_30_layer_call_fn_259220Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_flatten_4_layer_call_and_return_conditional_losses_259210]3?0
)?&
$?!
inputs?????????p
? "&?#
?
0??????????
? ~
*__inference_flatten_4_layer_call_fn_259204P3?0
)?&
$?!
inputs?????????p
? "????????????
R__inference_layer_normalization_31_layer_call_and_return_conditional_losses_258625d$%3?0
)?&
$?!
inputs?????????p@
? ")?&
?
0?????????p@
? ?
7__inference_layer_normalization_31_layer_call_fn_258578W$%3?0
)?&
$?!
inputs?????????p@
? "??????????p@?
R__inference_layer_normalization_34_layer_call_and_return_conditional_losses_259320\NO/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
7__inference_layer_normalization_34_layer_call_fn_259278ONO/?,
%?"
 ?
inputs?????????
? "??????????;
__inference_loss_fn_0_259363?

? 
? "? ;
__inference_loss_fn_1_259374E?

? 
? "? ;
__inference_loss_fn_2_259385V?

? 
? "? ?
C__inference_model_4_layer_call_and_return_conditional_losses_257408?$%cdefghijopklmnqrEFNOVW=?:
3?0
&?#
input_10?????????pd
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_257491?$%cdefghijopklmnqrEFNOVW=?:
3?0
&?#
input_10?????????pd
p

 
? "%?"
?
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_258016?$%cdefghijopklmnqrEFNOVW;?8
1?.
$?!
inputs?????????pd
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_258431?$%cdefghijopklmnqrEFNOVW;?8
1?.
$?!
inputs?????????pd
p

 
? "%?"
?
0?????????
? ?
(__inference_model_4_layer_call_fn_256637u$%cdefghijopklmnqrEFNOVW=?:
3?0
&?#
input_10?????????pd
p 

 
? "???????????
(__inference_model_4_layer_call_fn_257325u$%cdefghijopklmnqrEFNOVW=?:
3?0
&?#
input_10?????????pd
p

 
? "???????????
(__inference_model_4_layer_call_fn_257572s$%cdefghijopklmnqrEFNOVW;?8
1?.
$?!
inputs?????????pd
p 

 
? "???????????
(__inference_model_4_layer_call_fn_257629s$%cdefghijopklmnqrEFNOVW;?8
1?.
$?!
inputs?????????pd
p

 
? "???????????
H__inference_sequential_9_layer_call_and_return_conditional_losses_256053vklmnC?@
9?6
,?)
dense_36_input?????????p
p 

 
? ")?&
?
0?????????p
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_256067vklmnC?@
9?6
,?)
dense_36_input?????????p
p

 
? ")?&
?
0?????????p
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_259468nklmn;?8
1?.
$?!
inputs?????????p
p 

 
? ")?&
?
0?????????p
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_259525nklmn;?8
1?.
$?!
inputs?????????p
p

 
? ")?&
?
0?????????p
? ?
-__inference_sequential_9_layer_call_fn_255966iklmnC?@
9?6
,?)
dense_36_input?????????p
p 

 
? "??????????p?
-__inference_sequential_9_layer_call_fn_256039iklmnC?@
9?6
,?)
dense_36_input?????????p
p

 
? "??????????p?
-__inference_sequential_9_layer_call_fn_259398aklmn;?8
1?.
$?!
inputs?????????p
p 

 
? "??????????p?
-__inference_sequential_9_layer_call_fn_259411aklmn;?8
1?.
$?!
inputs?????????p
p

 
? "??????????p?
$__inference_signature_wrapper_258490?$%cdefghijopklmnqrEFNOVWA?>
? 
7?4
2
input_10&?#
input_10?????????pd"3?0
.
dense_39"?
dense_39??????????
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_258942vcdefghijopklmnqr7?4
-?*
$?!
inputs?????????p@
p 
? ")?&
?
0?????????p
? ?
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_259199vcdefghijopklmnqr7?4
-?*
$?!
inputs?????????p@
p
? ")?&
?
0?????????p
? ?
4__inference_transformer_block_9_layer_call_fn_258662icdefghijopklmnqr7?4
-?*
$?!
inputs?????????p@
p 
? "??????????p?
4__inference_transformer_block_9_layer_call_fn_258699icdefghijopklmnqr7?4
-?*
$?!
inputs?????????p@
p
? "??????????p