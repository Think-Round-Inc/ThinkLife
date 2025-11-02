'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Star, Heart, Send, CheckCircle, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';

interface ExperienceFormProps {
  onSuccess?: () => void;
}

export default function ExperienceForm({ onSuccess }: ExperienceFormProps) {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    title: '',
    content: '',
    rating: 0,
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleInputChange = (field: string, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleRatingClick = (rating: number) => {
    setFormData(prev => ({
      ...prev,
      rating: rating === prev.rating ? 0 : rating
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim() || !formData.content.trim()) {
      toast.error('Please fill in all required fields');
      return;
    }

    if (formData.content.length < 10) {
      toast.error('Please share more about your experience (at least 10 characters)');
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await fetch('/api/experiences', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: formData.name.trim(),
          email: formData.email.trim() || undefined,
          title: formData.title.trim() || undefined,
          content: formData.content.trim(),
          rating: formData.rating || undefined,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setIsSubmitted(true);
        toast.success('Thank you for sharing your experience! It will be reviewed before being published.');
        onSuccess?.();
        // Reset form after a delay
        setTimeout(() => {
          setFormData({
            name: '',
            email: '',
            title: '',
            content: '',
            rating: 0,
          });
          setIsSubmitted(false);
        }, 3000);
      } else {
        if (data.details) {
          // Show validation errors
          data.details.forEach((error: any) => {
            toast.error(`${error.field}: ${error.message}`);
          });
        } else {
          toast.error(data.error || 'Failed to submit your experience');
        }
      }
    } catch (error) {
      console.error('Submission error:', error);
      toast.error('Something went wrong. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isSubmitted) {
    return (
      <Card className="w-full max-w-2xl mx-auto border-green-200 bg-green-50/50">
        <CardContent className="pt-6">
          <div className="text-center space-y-4">
            <CheckCircle className="w-16 h-16 text-green-500 mx-auto" />
            <h3 className="text-2xl font-semibold text-green-800">Thank You!</h3>
            <p className="text-green-700">
              Your experience has been submitted successfully. It will be reviewed and published soon.
            </p>
            <div className="flex items-center justify-center gap-2 text-green-600">
              <Heart className="w-4 h-4 fill-current" />
              <span className="text-sm">We appreciate you sharing your story</span>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full max-w-2xl mx-auto shadow-lg border-[#D4C4B8]/50">
      <CardHeader className="bg-gradient-to-r from-[#5B2655] to-[#8B5A8F] text-white rounded-t-lg">
        <CardTitle className="text-2xl font-bold flex items-center gap-2">
          <Heart className="w-6 h-6" />
          Share Your Experience
        </CardTitle>
        <p className="text-white/90 mt-2">
          Help others by sharing your journey with ThinkLife. Your story matters.
        </p>
      </CardHeader>

      <CardContent className="p-6 space-y-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Name Field */}
          <div className="space-y-2">
            <Label htmlFor="name" className="text-[#5B2655] font-medium flex items-center gap-1">
              Your Name <span className="text-red-500">*</span>
            </Label>
            <Input
              id="name"
              type="text"
              placeholder="How should we display your name?"
              value={formData.name}
              onChange={(e) => handleInputChange('name', e.target.value)}
              className="border-[#D4C4B8] focus:border-[#5B2655] focus:ring-[#5B2655]"
              maxLength={100}
              required
            />
          </div>

          {/* Email Field */}
          <div className="space-y-2">
            <Label htmlFor="email" className="text-[#5B2655] font-medium">
              Email (Optional)
            </Label>
            <Input
              id="email"
              type="email"
              placeholder="your.email@example.com"
              value={formData.email}
              onChange={(e) => handleInputChange('email', e.target.value)}
              className="border-[#D4C4B8] focus:border-[#5B2655] focus:ring-[#5B2655]"
            />
            <p className="text-xs text-gray-500">
              We'll only use this for verification and won't display it publicly.
            </p>
          </div>

          {/* Title Field */}
          <div className="space-y-2">
            <Label htmlFor="title" className="text-[#5B2655] font-medium">
              Title (Optional)
            </Label>
            <Input
              id="title"
              type="text"
              placeholder="A brief title for your experience"
              value={formData.title}
              onChange={(e) => handleInputChange('title', e.target.value)}
              className="border-[#D4C4B8] focus:border-[#5B2655] focus:ring-[#5B2655]"
              maxLength={200}
            />
          </div>

          {/* Rating Field */}
          <div className="space-y-2">
            <Label className="text-[#5B2655] font-medium">
              How would you rate your experience? (Optional)
            </Label>
            <div className="flex items-center gap-1">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  type="button"
                  onClick={() => handleRatingClick(star)}
                  className="p-1 hover:scale-110 transition-transform"
                >
                  <Star
                    className={`w-6 h-6 transition-colors ${
                      star <= formData.rating
                        ? 'text-yellow-400 fill-current'
                        : 'text-gray-300 hover:text-yellow-200'
                    }`}
                  />
                </button>
              ))}
              {formData.rating > 0 && (
                <span className="ml-2 text-sm text-gray-600">
                  {formData.rating} star{formData.rating !== 1 ? 's' : ''}
                </span>
              )}
            </div>
          </div>

          {/* Content Field */}
          <div className="space-y-2">
            <Label htmlFor="content" className="text-[#5B2655] font-medium flex items-center gap-1">
              Your Experience <span className="text-red-500">*</span>
            </Label>
            <Textarea
              id="content"
              placeholder="Tell us about your experience with ThinkLife. How has it helped you? What do you appreciate most? Your story can inspire others..."
              value={formData.content}
              onChange={(e) => handleInputChange('content', e.target.value)}
              className="border-[#D4C4B8] focus:border-[#5B2655] focus:ring-[#5B2655] min-h-[120px] resize-y"
              maxLength={2000}
              required
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>Minimum 10 characters</span>
              <span>{formData.content.length}/2000</span>
            </div>
          </div>

          {/* Privacy Notice */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-blue-700">
                <p className="font-medium mb-1">Privacy & Moderation Notice</p>
                <p>
                  Your experience will be reviewed before being published to ensure it meets our community guidelines. 
                  We respect your privacy and will only display your name and experience publicly.
                </p>
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <Button
            type="submit"
            disabled={isSubmitting || !formData.name.trim() || !formData.content.trim()}
            className="w-full bg-gradient-to-r from-[#5B2655] to-[#8B5A8F] hover:from-[#8B5A8F] hover:to-[#5B2655] text-white font-medium py-3 transition-all duration-300"
          >
            {isSubmitting ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                Submitting...
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Send className="w-4 h-4" />
                Share My Experience
              </div>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
} 