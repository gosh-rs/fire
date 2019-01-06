initSidebarItems({"enum":[["LogLevel","An enum representing the available verbosity levels of the logger."]],"fn":[["create_dir","Recursively create a directory and all of its parent components if they are missing."],["err_msg","Constructs a `Fail` type from a string."],["glob","Find files with glob pattern"],["read_file","Read file content into string"],["remove_dir_all","Removes a directory at this path, after removing all its contents. Use carefully!"],["write_to_file","Write string to file"]],"macro":[["bail","Exits a function early with an `Error`."],["ensure","Exits a function early with an `Error` if the condition is not satisfied."],["error","Logs a message at the error level."],["format_err","Constructs an `Error` using the standard string interpolation syntax."],["info","Logs a message at the info level."],["log_enabled","Determines if a message logged at the specified level in that module will be logged."],["trace","Logs a message at the trace level."],["warn","Logs a message at the warn level."]],"mod":[["log","A lightweight logging facade."]],"struct":[["Error","The `Error` type, which can contain any failure."],["Verbosity","Easily add a `--verbose` flag to CLIs using Structopt"]],"trait":[["FromParallelIterator","`FromParallelIterator` implements the creation of a collection from a [`ParallelIterator`]. By implementing `FromParallelIterator` for a given type, you define how it will be created from an iterator."],["IndexedParallelIterator","An iterator that supports \"random access\" to its data, meaning that you can split it at arbitrary indices and draw data from those points."],["IntoParallelIterator","`IntoParallelIterator` implements the conversion to a [`ParallelIterator`]."],["IntoParallelRefIterator","`IntoParallelRefIterator` implements the conversion to a [`ParallelIterator`], providing shared references to the data."],["IntoParallelRefMutIterator","`IntoParallelRefMutIterator` implements the conversion to a [`ParallelIterator`], providing mutable references to the data."],["ParallelBridge","Conversion trait to convert an `Iterator` to a `ParallelIterator`."],["ParallelExtend","`ParallelExtend` extends an existing collection with items from a [`ParallelIterator`]."],["ParallelIterator","Parallel version of the standard iterator trait."],["ParallelSlice","Parallel extensions for slices."],["ParallelSliceMut","Parallel extensions for mutable slices."],["ParallelString","Parallel extensions for strings."],["ResultExt","Extension methods for `Result`."]],"type":[["CliResult","A handy alias for `Result` that carries a generic error type."],["Result",""]]});