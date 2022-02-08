import torch


def dynamic_partition_nd(data, partitions, num_partitions):
    """
    Args
        data: A Tensor.
        partitions:	A Tensor of type int32. Any shape. Indices in
            the range [0, num_partitions).
        num_partitions: An int that is >= 1. The number of partitions
            to output.
    Result:
        A list of num_partitions Tensor objects with the same type as data.
    """
    if isinstance(partitions, int):
        res = [torch.tensor([], dtype=data.dtype)
               for i in range(num_partitions)]

        res[partitions] = data
    else:
        res = []

        for i in range(num_partitions):
            print(data.shape, (partitions==i).nonzero().shape)
            res.append(data[(partitions == i).nonzero().unbind(1)])

    return res


def dynamic_partition(data, partitions, num_partitions):
    """
    Args
        data: A Tensor.
        partitions:	A Tensor of type int32. Any shape. Indices in
            the range [0, num_partitions).
        num_partitions: An int that is >= 1. The number of partitions
            to output.
    Result:
        A list of num_partitions Tensor objects with the same type as data.
    """
    if isinstance(partitions, int):
        res = [torch.tensor([], dtype=data.dtype)
               for i in range(num_partitions)]

        res[partitions] = data
    else:
        res = []

        for i in range(num_partitions):
            res.append(data[(partitions == i).nonzero().squeeze(-1)])

    return res


def dynamic_stitch(indices, data):
    """
    Args
        indices: A list of at least 1 Tensor objects with type int32.
        data: A list with the same length as indices of Tensor objects with
             the same type.
    Returns
        A Tensor. Has the same type as data.
    """
    dim_0 = int(max([torch.max(idx) if idx.shape[0] != 0
                     else 0 for idx in indices]) + 1)
    shape = torch.Size([dim_0] + list(data[0].shape[indices[0].ndim:]))
    tensor = torch.empty(shape, dtype=data[0].dtype)

    for i in range(len(indices)):
        tensor[indices[i]] = data[i]

    return tensor


def unsorted_segment_sum_1d(data: torch.Tensor, segment_ids: torch.Tensor,
                            num_segments: int):
    tensor = torch.zeros([num_segments], dtype=data.dtype, device=data.device)
    tensor = tensor.scatter_add(0, segment_ids, data)

    return tensor


def unsorted_segment_sum_2d(data: torch.Tensor, segment_ids: torch.Tensor,
                            num_segments: int):
    batch_size = int(data.shape[0])
    hidden_size = int(data.shape[1])

    segment_ids = segment_ids.repeat_interleave(hidden_size).view(batch_size,
                                                                  hidden_size)
    tensor = torch.zeros([num_segments, hidden_size], device=data.device,
                          dtype=data.dtype)
    tensor = tensor.scatter_add(0, segment_ids, data)

    return tensor


def top_k_softmax(logits, k, n):
    logits_shape = logits.shape
    logits = torch.reshape(logits, [-1, n])

    top_logits, top_indices = torch.topk(logits, k=min(k + 1, n))

    top_k_logits = top_logits[:, :k]
    top_k_indices = top_indices[:, :k]

    probs = torch.softmax(top_k_logits, dim=-1)
    batch = int(top_k_logits.shape[0])
    k = int(top_k_logits.shape[1])

    # Flat to 1D
    indices_flat = torch.reshape(top_k_indices, [-1])
    indices_flat = indices_flat + (
        torch.arange(batch * k, device=logits.device) // k * n)
    ret_flat = unsorted_segment_sum_1d(torch.reshape(probs, [-1]),
                                       indices_flat.long(), batch * n)
    return torch.reshape(ret_flat, logits_shape)
