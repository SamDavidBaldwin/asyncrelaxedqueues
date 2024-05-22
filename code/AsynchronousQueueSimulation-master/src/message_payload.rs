use memoffset::offset_of;
use mpi::datatype::{Equivalence, UserDatatype};
use mpi::Rank;
use std::cmp::Ordering;
use std::mem::size_of;
use std::{fmt, usize};

const MAX_BUFFER_SIZE: usize = 32; // Upper limit on the number of processes in the system

#[derive(Copy, Clone)]
pub(crate) struct VectorClock {
    pub clock: [i32; MAX_BUFFER_SIZE],
    pub size: usize, // Tracks the number of active elements in `clock`
}

impl VectorClock {
    pub fn new(size: i32) -> Self {
        VectorClock {
            clock: [0; MAX_BUFFER_SIZE],
            size: size as usize,
        }
    }

    pub fn compare(&self, other: &Self) -> Ordering {
        for i in 0..self.size {
            if self.clock[i] != other.clock[i] {
                return self.clock[i].cmp(&other.clock[i]);
            }
        }
        Ordering::Equal
    }
}

impl PartialOrd for VectorClock {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.compare(other))
    }
}

impl PartialEq for VectorClock {
    fn eq(&self, other: &Self) -> bool {
        self.compare(other) == Ordering::Equal
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self {
            clock: [0; MAX_BUFFER_SIZE], // Manually initializing the array
            size: 0,                     // Default logical size is 0
        }
    }
}

impl fmt::Debug for VectorClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Start with the struct name
        write!(f, "VectorClock: [")?;

        // Iterate only up to `size` to show the active elements
        if self.size > 0 {
            for i in 0..self.size - 1 {
                write!(f, "{}, ", self.clock[i])?;
            }
            // Write the last element without a trailing comma
            write!(f, "{}", self.clock[self.size - 1])?;
        }

        // Close the bracket
        write!(f, "]")
    }
}

unsafe impl Equivalence for VectorClock {
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        let mut displacements = Vec::new();
        for i in (0..MAX_BUFFER_SIZE + 2).rev() {
            displacements.push((i * size_of::<i32>()) as mpi::Address);
        }
        displacements.push(offset_of!(VectorClock, size) as mpi::Address);

        let mut datatypes = vec![i32::equivalent_datatype(); MAX_BUFFER_SIZE + 2];
        datatypes.push(usize::equivalent_datatype());

        UserDatatype::structured(
            &vec![1; MAX_BUFFER_SIZE + 2 + 1],
            &displacements,
            &datatypes,
        )
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct MessagePayload {
    pub value: i32,
    pub message: i32,
    pub invoker: Rank,
    pub sender: Rank,
    pub receiver: Rank,
    pub time_stamp: VectorClock,
}

impl MessagePayload {
    pub fn new(
        msg: i32,
        val: i32,
        inv: Rank,
        sender: Rank,
        receiver: Rank,
        ts: VectorClock,
    ) -> Self {
        MessagePayload {
            message: msg,
            value: val,
            invoker: inv,
            sender: sender,
            receiver: receiver,
            time_stamp: ts,
        }
    }

    pub fn with_msg(msg: i32) -> Self {
        MessagePayload {
            message: msg,
            value: -1,
            invoker: -1,
            sender: -1,
            receiver: -1,
            time_stamp: VectorClock::default(),
        }
    }
}

unsafe impl Equivalence for MessagePayload {
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        let displacements = [
            offset_of!(MessagePayload, message) as mpi::Address,
            offset_of!(MessagePayload, value) as mpi::Address,
            offset_of!(MessagePayload, invoker) as mpi::Address,
            offset_of!(MessagePayload, sender) as mpi::Address,
            offset_of!(MessagePayload, receiver) as mpi::Address,
            offset_of!(MessagePayload, time_stamp) as mpi::Address,
        ];

        UserDatatype::structured(
            &[1, 1, 1, 1, 1, 1], // One block of each type
            &displacements,
            &[
                i32::equivalent_datatype(),                  // Datatype for message
                i32::equivalent_datatype(),                  // Datatype for value
                Rank::equivalent_datatype(),                 // Datatype for invoker
                Rank::equivalent_datatype(),                 // Datatype for sender
                Rank::equivalent_datatype(),                 // Datatype for receiver
                VectorClock::equivalent_datatype().as_ref(), // Datatype for time_stamp
            ],
        )
    }
}
