#ifndef __UTILS_H__
#define __UTILS_H__

#define DISALLOW_DEFAULT(ClassName) \
  ClassName(){}

#define DISALLOW_COPY_AND_ASSIGN(ClassName) \
  ClassName(const ClassName&); \
  void operator=(const ClassName&)

#endif
