"""
pipeline partition
"""
from bisect import bisect_left
from math import floor

def partition_imbalanced(num_layers, num_parts, eps=1e-3):
      """
      partition imbalanced.
      first and last stages contain less layers,
      other stages contain more layers
      """
      # First check for the trivial edge case
      if num_layers <= num_parts:
          parts = partition_uniform(num_layers, num_parts)
      else:
          parts = [0] * (num_parts + 1)
          if num_layers % num_parts == 0:
              parts_count = [num_layers // num_parts] * num_parts
          else:
              parts_count = [num_layers // num_parts + 1] * num_parts
              res = num_layers - parts_count[0] * (num_parts - 2)
              parts_count[0] = res // 2
              if res % 2 == 0:
                  parts_count[num_parts - 1] = parts_count[0]
              else:
                  parts_count[num_parts - 1] = parts_count[0] + 1
          for i in range(1, len(parts_count) + 1):
              parts[i] = parts[i - 1] + parts_count[i - 1]

      return parts_count, parts


def partition_balanced(num_layers, num_parts, eps=1e-3):
    """
    partition balanced.
    """
    # First check for the trivial edge case
    if num_layers <= num_parts:
        parts = partition_uniform(num_layers, num_parts)
    else:
        weights = [1] * num_layers
        weights_ = prefix_sum_inc(weights)

        # Find the smallest bottleneck (weight of heaviest partition)
        bottleneck = _rb_partition_balanced(weights_, num_parts, eps=eps)

        # Now compute that partitioning
        parts, success = _lprobe(weights_, num_parts, bottleneck)
        assert success

    parts_count = [0] * num_parts

    for i in range(1, len(parts)):
        parts_count[i - 1] = parts[i] - parts[i - 1]

    return parts_count, parts


def partition_customized(partition_split):
    """
    user customized partition.
    """
    parts_count = list(map(int, partition_split.split(",")))
    parts = [0] * (len(parts_count) + 1)
    for i in range(1, len(parts)):
        parts[i] = parts[i - 1] + parts_count[i - 1]
    return parts_count, parts


def partition_uniform(num_items, num_parts):
    """
    partition uniform.
    """
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    chunksize = floor(num_items / num_parts)
    for p in range(num_parts):
        parts[p] = min(chunksize * p, num_items)
    parts[num_parts] = num_items
    return parts


def prefix_sum_inc(weights):
    """ Compute an inclusive prefix sum.

    Example:
        >>> prefix_sum_inc([3,4,5])
        [3, 7, 12]
    """
    weights_ = [w for w in weights]
    for x in range(1, len(weights_)):
        weights_[x] += weights_[x - 1]
    return weights_


def _rb_partition_balanced(weights, num_parts, eps):
    total_weight = weights[-1]
    lower = total_weight / num_parts  # best case heaviest partition
    upper = total_weight  # worst case heaviest partition

    # Do a binary search for the best partitioning
    while upper > lower + eps:
        mid = lower + ((upper - lower) / 2)
        parts, success = _lprobe(weights, num_parts, mid)
        if success:
            upper = mid
        else:
            lower = mid + eps
    return upper


def _lprobe(weights, num_parts, bottleneck):
    num_items = len(weights)
    total_weight = weights[-1]

    # initialize partitioning
    parts = [0] * (num_parts + 1)
    for p in range(1, num_parts + 1):
        parts[p] = num_items

    bsum = bottleneck  # running sum of target weight for pth partition
    chunksize = num_items // num_parts
    step = chunksize
    for p in range(1, num_parts):
        # Jump to the next bucket
        while (step < num_items) and (weights[step] < bsum):
            step += chunksize

        # Find the end index of partition p
        parts[p] = bisect_left(weights, bsum, lo=step - chunksize, hi=min(step, num_items))
        # Nothing more to partition, return early
        if parts[p] == num_items:
            # See if the current partition is overweight.
            part_size = weights[-1] - weights[parts[p - 1]]
            return parts, part_size < bottleneck

        # Next partition target
        bsum = weights[parts[p] - 1] + bottleneck

    return parts, bsum >= total_weight

def partition_balanced_optimal(num_layers, num_parts, eps=1e-3):
    """
    partition balanced optimal func.
    """
    if num_layers <= num_parts:
        parts = partition_uniform(num_layers, num_parts)
        parts_count = [0] * num_parts
        for i in range(1, len(parts)):
            parts_count[i - 1] = parts[i] - parts[i - 1]
    else:
        layers_per_part = num_layers // num_parts

        parts_count = [layers_per_part] * num_parts
        left_parts = num_layers - layers_per_part * num_parts
        if num_parts > 2 and left_parts > 0:
            for i in range(1, num_parts - 1):
                parts_count[i] += 1
                left_parts -= 1
                if left_parts <= 0:
                    break
        if left_parts > 0:
            parts_count[0] += 1
        parts = [0] + parts_count
        for i in range(1, len(parts)):
            parts[i] += parts[i - 1]
    return parts_count, parts



