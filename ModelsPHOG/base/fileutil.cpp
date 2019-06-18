/*
   Copyright 2015 Software Reliability Lab, ETH Zurich

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#include "base/fileutil.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdio.h>  /* defines FILENAME_MAX */
// #define WINDOWS  /* uncomment this line to use it for windows.*/
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#include<iostream>

std::string GetCurrentWorkingDir( void ) {
    char buff[FILENAME_MAX];
    GetCurrentDir( buff, FILENAME_MAX );
    std::string current_working_dir(buff);
    return current_working_dir;
}

void ReadFileToStringOrDie(const char* filename, std::string* r) {
  r->clear();
  LOG(INFO) << GetCurrentWorkingDir();
  FILE* f = fopen(filename, "rt");
  CHECK(f != NULL) << "Could not open " << filename << " for reading.";
  char buf[4096];
  while (!feof(f)) {
    if (fgets(buf, 4096, f) == buf) {
      r->append(buf);
    } else {
      if (!feof(f))
        LOG(FATAL) << "Read error from " << filename;
    }
  }
  fclose(f);
}

void WriteStringToFileOrDie(const char* filename, const std::string& s) {
  FILE* f = fopen(filename, "wt");
  CHECK(f != NULL) << "Could not open " << filename << " for writing.";
  fprintf(f, "%s", s.c_str());
  fclose(f);
}

bool FileExists(const char* filename) {
  struct stat stat_info;
  return stat(filename, &stat_info) == 0 && !S_ISDIR(stat_info.st_mode);
}
