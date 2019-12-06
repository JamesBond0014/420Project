#        # Extra set of code in the case of "less confidence"
#        if max_val < tolerance:
#            alt = cv2.matchTemplate(img,STARTING_TEMPLATE,method)
#            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(alt)
#        elif max_val < 0.99:
#            print("special case for %d" % step)
#            alt = cv2.matchTemplate(img,STARTING_TEMPLATE,method)
#            # Assume max_val shouldn't dip below 0.93:
#            res_weight = (max_val - tolerance)/(0.06)
#            alt_weight = 1-res_weight
#            final = res*res_weight + alt*alt_weight
#            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(final)
#